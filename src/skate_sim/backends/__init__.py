def create_backend(name: str, **kwargs):
    if name == "mujoco":
        from .mujoco import MujocoBackend
        return MujocoBackend(**kwargs)
    if name == "mjlab":
        from .mjlab import MjlabBackend
        return MjlabBackend(**kwargs)
    raise ValueError(f"unknown backend: {name}")
