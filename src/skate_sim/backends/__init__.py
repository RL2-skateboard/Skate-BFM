def create_backend(name: str, **kwargs):
    if name == "mujoco":
        from .mujoco import MujocoBackend
        return MujocoBackend(**kwargs)
    if name == "mjlab":
        from .mjlab import MjlabBackend
        backend = MjlabBackend(**kwargs)
        return backend
    if name == "isaacsim":
        from .isaacsim import IsaacSimBackend
        return IsaacSimBackend(**kwargs)
    raise ValueError(f"unknown backend: {name}")
