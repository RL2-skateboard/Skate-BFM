# Copyright 2025 The Newton Developers
# Adapted contact field gather design; repository-specific ordering and dispatch.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed
# under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
# CONDITIONS OF ANY KIND, either express or implied. See the License for the
# specific language governing permissions and limitations under the License.

"""Deterministic CUDA ordering for the pinned MuJoCo Warp rigid scene backend.

Contact schema-wide permutation follows the design of MuJoCo Warp PR #1775
(e916b8e8e9c50b69064da0101551c17da97731a4), Apache-2.0, Newton Developers.
Stable radix passes use unpacked keys, with no changes to contact payloads.
World-local scalar reductions retain ascending logical-item order on CUDA.
"""

import types
import ast
import linecache
import textwrap
import inspect
import hashlib
import importlib.metadata
from pathlib import Path
from functools import lru_cache

import warp as wp
from warp._src.utils import radix_sort_pairs

STRATEGY = "rigid-world-interleaved-radix-v4"
# Pinned dependency signatures: every listed reduction writes only its own world.
WORLD_KERNEL_TIDS = {
    "_update_constraint_init_qfrc_constraint_sparse__locals__kernel": ("worldid", "efcid"),
    "_JTDAJ_sparse__locals__kernel": ("worldid", "slot", "lane"),
    "_update_constraint_efc__locals__kernel": ("worldid", "efcid"),
    "_update_gradient_h_incremental_sparse__locals__kernel": ("worldid", "slot", "lane"),
    "_update_gradient_grad__locals__kernel": ("worldid", "dofid"),
    "_friction_dof__locals__kernel": ("worldid", "dofid"),
    "_equality_joint__locals__kernel": ("worldid", "eqjntid"),
    "_limit_slide_hinge__locals__kernel": ("worldid", "jntlimitedid"),
    "_subtree_com_acc": ("worldid", "nodeid"),
    "_crb_accumulate": ("worldid", "nodeid"),
    "_cfrc_backward": ("worldid", "nodeid"),
}
SLOT_PARAMETERS = {
    "_JTDAJ_sparse__locals__kernel": "groups_per_world",
    "_update_gradient_h_incremental_sparse__locals__kernel": "slots_per_world",
}
SERIAL_KERNEL_KEY = "_efc_contact_init__locals__kernel"
HELPER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


@lru_cache(None)
def _world_interleaved(kernel):
    """Move world to the fastest logical axis, preserving the kernel body.

    With block_dim=1 and max_blocks=nworld, CUDA grid-stride block w visits
    only world w, in the original ascending item/lane order. Each specialization
    retains its own closure globals and unique Warp module; no global WP patch.
    """
    expected = WORLD_KERNEL_TIDS.get(kernel.key)
    if expected is None:
        raise RuntimeError(f"unverified world reduction kernel: {kernel.key}")
    tree = ast.parse(textwrap.dedent(inspect.getsource(kernel.func)))
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise RuntimeError(f"unverified kernel source: {kernel.key}")
    function = tree.body[0]
    bindings = [node for node in ast.walk(function)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "wp"
                and node.func.attr == "tid"]
    assignments = [node for node in ast.walk(function)
                   if isinstance(node, ast.Assign) and node.value in bindings]
    if len(bindings) != 1 or len(assignments) != 1 or len(assignments[0].targets) != 1:
        raise RuntimeError(f"unverified tid binding: {kernel.key}")
    target = assignments[0].targets[0]
    if (not isinstance(target, ast.Tuple) or
            any(not isinstance(item, ast.Name) for item in target.elts) or
            tuple(item.id for item in target.elts) != expected):
        raise RuntimeError(f"changed tid signature: {kernel.key}")
    target.elts = target.elts[1:] + target.elts[:1]
    function.decorator_list = []
    function.name = "world_interleaved_" + kernel.key
    source = ast.unparse(tree) + "\n"
    filename = "<skate_world_" + hashlib.sha256(source.encode()).hexdigest() + ".py>"
    # inspect/Warp and traceback can retrieve exactly the transformed source.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(kernel.func.__globals__)
    namespace.update(inspect.getclosurevars(kernel.func).nonlocals)
    exec(compile(source, filename, "exec"), namespace)
    return wp.kernel(namespace[function.name], module="unique", enable_backward=False)


class _LaunchDelegate:
    """Adapt only the pinned allowlist in private dependency dispatch globals."""
    def __init__(self, original):
        self.original = original

    def __getattr__(self, name):
        return getattr(self.original, name)

    def launch(self, kernel, *args, **kwargs):
        if kernel.key in WORLD_KERNEL_TIDS:
            dims = kwargs.get("dim")
            if (args or not isinstance(dims, (tuple, list)) or
                    len(dims) != len(WORLD_KERNEL_TIDS[kernel.key]) or
                    any(not isinstance(size, int) or size < 0 for size in dims) or dims[0] < 1):
                raise RuntimeError(f"changed launch dimensions: {kernel.key}: {dims}")
            dims = tuple(dims)
            if kernel.key in SLOT_PARAMETERS:
                parameter = SLOT_PARAMETERS[kernel.key]
                parameters = tuple(inspect.signature(kernel.func).parameters)
                inputs = list(kwargs.get("inputs", ()))
                if parameter not in parameters:
                    raise RuntimeError(f"changed reduction stride signature: {kernel.key}")
                index = parameters.index(parameter)
                if index >= len(inputs) or inputs[index] != dims[1]:
                    raise RuntimeError(f"changed reduction slot stride: {kernel.key}")
                # A scalar world needs one slot. The kernel's original grid-stride
                # loop still visits every constraint group/change, in fixed order.
                inputs[index] = 1
                kwargs["inputs"] = inputs
                dims = (dims[0], 1, dims[2])
            kernel = _world_interleaved(kernel)
            kwargs.update(dim=dims[1:] + dims[:1], block_dim=1, max_blocks=dims[0])
        elif kernel.key == SERIAL_KERNEL_KEY:
            # Contact IDs share global ordering and per-world row allocation.
            kwargs.update(block_dim=1, max_blocks=1)
        return self.original.launch(kernel, *args, **kwargs)


def configure_launch_order() -> None:
    supported = {"mujoco-warp": "3.11.0", "warp-lang": "1.17.0"}
    actual = {name: importlib.metadata.version(name) for name in supported}
    if actual != supported:
        raise RuntimeError(f"unverified CUDA ordering API versions: {actual}; expected {supported}")
    from mujoco_warp._src import constraint, solver, smooth
    dispatch_names = {
        constraint: ("make_constraint",),
        solver: ("_update_constraint", "_update_gradient", "_update_gradient_incremental"),
        smooth: ("com_pos", "crb", "_rne_cfrc_backward"),
    }
    for module, names in dispatch_names.items():
        if getattr(module, "_skate_launch_order", False):
            continue
        # Kernel factories retain the actual Warp module in their globals.
        # Only these Python launch dispatchers receive a private delegate.
        dispatch_globals = dict(vars(module))
        dispatch_globals["wp"] = _LaunchDelegate(wp)
        replacements = {}
        for name in names:
            function = getattr(module, name)
            raw = inspect.unwrap(function)
            clone = types.FunctionType(raw.__code__, dispatch_globals, raw.__name__,
                                       raw.__defaults__, raw.__closure__)
            clone.__kwdefaults__ = raw.__kwdefaults__
            clone.__annotations__ = raw.__annotations__
            if raw is not function:
                clone = module.event_scope(clone)
            replacements[name] = clone
        dispatch_globals.update(replacements)
        for name, clone in replacements.items():
            setattr(module, name, clone)
        module._skate_launch_order = True


@wp.kernel(enable_backward=False)
def _contact_ids(permutation: wp.array[int]):
    cid = wp.tid()
    permutation[cid] = cid


@wp.kernel(enable_backward=False)
def _contact_keys(nacon: wp.array[int], world: wp.array[int],
                  geom: wp.array[wp.vec2i], gcid: wp.array[int],
                  permutation: wp.array[int], keys: wp.array[wp.int64], component: int):
    cid = wp.tid()
    original = permutation[cid]
    # All valid keys are int32; the wider sentinel always sorts after them.
    key = wp.int64(9223372036854775807)
    if original < wp.min(nacon[0], permutation.shape[0]):
        if component == 0:
            key = wp.int64(gcid[original])
        elif component == 1:
            key = wp.int64(geom[original][1])
        elif component == 2:
            key = wp.int64(geom[original][0])
        else:
            key = wp.int64(world[original])
    keys[cid] = key


@lru_cache(None)
def _gather(dtype, ndim):
    if ndim == 1:
        @wp.kernel(module="unique", enable_backward=False)
        def kernel(nacon: wp.array[int], permutation: wp.array[int],
                   source: wp.array(dtype=dtype), target: wp.array(dtype=dtype)):
            cid = wp.tid()
            if cid < wp.min(nacon[0], permutation.shape[0]):
                target[cid] = source[permutation[cid]]
    elif ndim == 2:
        @wp.kernel(module="unique", enable_backward=False)
        def kernel(nacon: wp.array[int], permutation: wp.array[int],
                   source: wp.array2d(dtype=dtype), target: wp.array2d(dtype=dtype)):
            cid, col = wp.tid()
            if cid < wp.min(nacon[0], permutation.shape[0]):
                target[cid, col] = source[permutation[cid], col]
    else:
        raise RuntimeError(f"unsupported Contact field dimension {ndim}")
    return kernel


def install_contact_order(sim) -> None:
    """Install before graph capture/reset; keep all permutation buffers alive."""
    model, data = sim.wp_model, sim.wp_data
    if model.nflex:
        raise RuntimeError("deterministic contact ordering supports rigid scenes only")
    previous = model.callback.contactfilter
    values = wp.empty(2 * data.naconmax, dtype=int, device=sim.wp_device)
    keys = wp.empty(2 * data.naconmax, dtype=wp.int64, device=sim.wp_device)
    permutation = values[:data.naconmax]
    fields = []
    for name in data.contact.__dataclass_fields__:
        field = getattr(data.contact, name)
        if field.size:
            fields.append((field, wp.empty_like(field), _gather(field.dtype, field.ndim)))

    def order_contacts(m, d):
        if previous:
            previous(m, d)
        wp.launch(_contact_ids, dim=d.naconmax, device=sim.wp_device, inputs=[permutation])
        # Stable LSD passes preserve original cid for exact duplicate keys.
        for component in range(4):
            wp.launch(_contact_keys, dim=d.naconmax, device=sim.wp_device,
                      inputs=[d.nacon, d.contact.worldid, d.contact.geom,
                              d.contact.geomcollisionid, permutation, keys, component])
            radix_sort_pairs(keys, values, d.naconmax)
        for target, source, kernel in fields:
            wp.copy(source, target)
            wp.launch(kernel, dim=target.shape, device=sim.wp_device,
                      inputs=[d.nacon, permutation, source], outputs=[target])

    model.callback.contactfilter = order_contacts
    sim._skate_contact_order = (order_contacts, permutation, values, keys, fields)
    sim.create_graph()


def numerics_manifest(sim) -> dict:
    return {
        "strategy": STRATEGY,
        "helper_sha256": HELPER_SHA256,
        "device_arch": sim.wp_device.arch,
        "state_dtype": str(sim.wp_data.qpos.dtype),
        "dispatch_isolation": "seven-private-dispatchers-original-factory-globals-v2",
        "contact_sort": "world-geom0-geom1-geomcollisionid-stable-radix-all-fields-v2",
        "world_kernel_launches": {name: {"block_dim": 1, "max_blocks": sim.wp_data.nworld,
                                          "tid": list(tids[1:] + tids[:1])}
                                  for name, tids in WORLD_KERNEL_TIDS.items()},
        "reduction_slots": {name: {parameter: 1} for name, parameter in SLOT_PARAMETERS.items()},
        "scalar_kernel_launches": {SERIAL_KERNEL_KEY: {"block_dim": 1, "max_blocks": 1}},
        "versions": {name: importlib.metadata.version(name)
                     for name in ("mujoco", "mujoco-warp", "warp-lang", "mjlab")},
    }
