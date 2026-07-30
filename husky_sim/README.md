# HUSKY Simulator

This directory contains the HUSKY simulator boundary used by Skate-BFM.

- `upstream/`: pinned HUSKY source, assets, datasets, and training code.
- `src/skate_husky/lite_env.py`: project-owned headless runtime for integration
  tests.

The submodule points to
[`TeleHuman/humanoid_skateboarding`](https://github.com/TeleHuman/humanoid_skateboarding)
at commit `d93833e80deff7f927c0b80ef9c435d8b5c488fe`. Run

```bash
git submodule update --init --depth 1
```

after cloning this repository. See
the upstream repository for the original documentation.

HUSKY is distributed under CC BY-NC 4.0. The original license is retained in
[`upstream/LICENSE-CC-BY-NC-4.0.md`](upstream/LICENSE-CC-BY-NC-4.0.md).
