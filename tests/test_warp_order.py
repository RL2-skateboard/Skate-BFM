"""CPU permutation edge cases; physical GPU integration is validated separately."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
try:
    import warp as wp
    from skate_sim.backends.warp import install_contact_order, _LaunchDelegate, _world_interleaved
except ModuleNotFoundError:
    wp = None


@unittest.skipIf(wp is None, "optional Warp dependency is unavailable")
class ContactOrderTest(unittest.TestCase):
    def test_all_payloads_follow_permutation_and_ties_preserve_contacts(self):
        wp.init()
        contact = SimpleNamespace(
            worldid=wp.array([1, 0, 0, 0], dtype=int, device="cpu"),
            geom=wp.array([[2, 3], [4, 5], [2, 3], [2, 3]], dtype=wp.vec2i, device="cpu"),
            geomcollisionid=wp.array([0, 0, 1, 1], dtype=int, device="cpu"),
            dist=wp.array([10., 20., 30., 40.], dtype=float, device="cpu"),
            efc_address=wp.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=int, device="cpu"),
        )
        contact.__dataclass_fields__ = dict.fromkeys(vars(contact))
        data = SimpleNamespace(naconmax=4, nacon=wp.array([4], dtype=int, device="cpu"), contact=contact)
        model = SimpleNamespace(nflex=0, callback=SimpleNamespace(contactfilter=None))
        sim = SimpleNamespace(wp_model=model, wp_data=data, wp_device="cpu", create_graph=lambda: None)
        install_contact_order(sim)
        model.callback.contactfilter(model, data)
        np.testing.assert_array_equal(contact.dist.numpy(), [30, 40, 20, 10])
        np.testing.assert_array_equal(contact.efc_address.numpy(), [[5, 6], [7, 8], [3, 4], [1, 2]])
        np.testing.assert_array_equal(data.nacon.numpy(), [4])
        data.nacon.zero_()
        model.callback.contactfilter(model, data)
        np.testing.assert_array_equal(contact.dist.numpy(), [30, 40, 20, 10])

    def test_launch_delegate_leaves_other_kernels_unchanged(self):
        calls = []
        original = SimpleNamespace(launch=lambda *a, **k: calls.append(k))
        delegate = _LaunchDelegate(original)
        delegate.launch(SimpleNamespace(key="unrelated"), dim=2, block_dim=128)
        delegate.launch(SimpleNamespace(key="_efc_contact_init__locals__kernel"), dim=2)
        self.assertEqual(calls, [dict(dim=2, block_dim=128), dict(dim=2, block_dim=1, max_blocks=1)])

    def test_dispatch_keeps_real_warp_module_in_kernel_factory_globals(self):
        from skate_sim.backends.warp import configure_launch_order
        from mujoco_warp._src import constraint, solver
        configure_launch_order()
        self.assertIs(constraint.wp, wp)
        self.assertIs(solver.wp, wp)
        self.assertIsInstance(constraint.make_constraint.__wrapped__.__globals__["wp"], _LaunchDelegate)
        self.assertIsInstance(solver._update_constraint.__globals__["wp"], _LaunchDelegate)
        configure_launch_order()
        self.assertIs(constraint.wp, wp)

    def test_world_launch_rotates_dimensions_and_rejects_changed_signature(self):
        from skate_sim.backends import warp as helper
        calls = []
        delegate = _LaunchDelegate(SimpleNamespace(launch=lambda k, **kw: calls.append((k, kw))))
        original = SimpleNamespace(key="_update_constraint_efc__locals__kernel")
        replacement = object()
        with patch.object(helper, "_world_interleaved", return_value=replacement):
            delegate.launch(original, dim=(5, 13), inputs=["payload"])
        self.assertEqual(calls, [(replacement, dict(dim=(13, 5), inputs=["payload"],
                                                   block_dim=1, max_blocks=5))])
        for invalid in (5, (5, 13, 7), (5, -1)):
            with self.assertRaises(RuntimeError):
                delegate.launch(original, dim=invalid)

    def test_transformed_specializations_keep_independent_closure_values(self):
        from mujoco_warp._src import solver
        for factory, name in ((solver._JTDAJ_sparse, "COMPACT"),
                              (solver._update_constraint_efc, "TRACK_CHANGES")):
            a, b = _world_interleaved(factory(False)), _world_interleaved(factory(True))
            self.assertIsNot(a, b)
            self.assertFalse(a.func.__globals__[name])
            self.assertTrue(b.func.__globals__[name])
            self.assertIs(a.func.__globals__["wp"], wp)

    def test_radix_full_int32_keys_and_inactive_capacity_preserve_payload(self):
        wp.init()
        rng = np.random.default_rng(72)
        capacity, count = 257, 173
        world = rng.integers(0, 5, capacity, dtype=np.int32)
        geom = rng.integers(-3, 4, (capacity, 2), dtype=np.int32)
        gcid = rng.integers(-3, 4, capacity, dtype=np.int32)
        gcid[0], gcid[1] = np.iinfo(np.int32).min, np.iinfo(np.int32).max
        expected = sorted(range(count), key=lambda i: (world[i], *geom[i], gcid[i], i))
        contact = SimpleNamespace(worldid=wp.array(world, dtype=int, device="cpu"),
                                  geom=wp.array(geom, dtype=wp.vec2i, device="cpu"),
                                  geomcollisionid=wp.array(gcid, dtype=int, device="cpu"),
                                  payload=wp.array(np.arange(capacity), dtype=int, device="cpu"))
        contact.__dataclass_fields__ = dict.fromkeys(vars(contact))
        data = SimpleNamespace(naconmax=capacity, nacon=wp.array([count], dtype=int, device="cpu"),
                               contact=contact)
        model = SimpleNamespace(nflex=0, callback=SimpleNamespace(contactfilter=None))
        sim = SimpleNamespace(wp_model=model, wp_data=data, wp_device="cpu", create_graph=lambda: None)
        install_contact_order(sim)
        model.callback.contactfilter(model, data)
        np.testing.assert_array_equal(contact.payload.numpy()[:count], expected)
        np.testing.assert_array_equal(contact.payload.numpy()[count:], np.arange(count, capacity))
        np.testing.assert_array_equal(contact.worldid.numpy()[:count], world[expected])
        np.testing.assert_array_equal(contact.geom.numpy()[:count], geom[expected])
        np.testing.assert_array_equal(contact.geomcollisionid.numpy()[:count], gcid[expected])
        np.testing.assert_array_equal(data.nacon.numpy(), [count])

    def test_reduction_slot_and_stride_collapse_together(self):
        import inspect
        from mujoco_warp._src import solver
        from skate_sim.backends import warp as helper
        calls = []
        delegate = _LaunchDelegate(SimpleNamespace(launch=lambda k, **kw: calls.append(kw)))
        kernel = solver._JTDAJ_sparse(False)
        index = tuple(inspect.signature(kernel.func).parameters).index("groups_per_world")
        inputs = [None] * (index + 1)
        inputs[index] = 13
        with patch.object(helper, "_world_interleaved", return_value=object()):
            delegate.launch(kernel, dim=(5, 13, 32), inputs=inputs)
        self.assertEqual(calls[0]["dim"], (1, 32, 5))
        self.assertEqual(calls[0]["inputs"][index], 1)
        self.assertEqual(inputs[index], 13)
        inputs[index] = 12
        with self.assertRaises(RuntimeError):
            delegate.launch(kernel, dim=(5, 13, 32), inputs=inputs)
