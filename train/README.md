# `train` Branch

This branch owns model-training code, training configuration, data preparation,
checkpoints, and learned motion-library development. It is intentionally
separate from `main`, which contains the BFM0-HUSKY integration and formal H1
evaluation records.

Training datasets belong under [`train/dataset/`](dataset/). Generated
checkpoints, exports, and motion-library artifacts belong under
[`model/motion_library/`](../model/motion_library/).

Training records start from zero in:

- [`train_log.md`](train_log.md): dated work log.
- [`train_res.md`](train_res.md): parameters, checkpoints, metrics, and results.

Large datasets and generated model files are local artifacts and are ignored by
Git.
