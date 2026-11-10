# Third-party software and artifacts

This repository contains orchestration and adapters for independently
developed models. It does not place those models, their pretrained weights, or
their associated data under a repository-wide license. The terms published by
each upstream project or artifact provider control their use.

In particular:

- [PRESAGE](https://github.com/Genentech/PRESAGE) is fetched from a pinned
  upstream revision when its environment is built. The corresponding recipe
  identifies its license as `LicenseRef-Genentech-NonCommercial-1.0`.
- [scLAMBDA](https://github.com/gefeiwang/scLAMBDA) is fetched from a pinned
  upstream revision when its environment is built. The corresponding recipe
  identifies its license as `GPL-3.0-only`.
- GEARS, scGPT, STATE, TabICL and their transitive dependencies are installed
  from their upstream distributions or pinned source revisions. Their upstream
  licenses continue to apply.
- Pretrained checkpoints, gene embeddings, model caches and datasets are
  downloaded separately. Their source-specific terms continue to apply and
  they are not redistributed by this repository.

The model-environment recipes and lock files record the sources and revisions
used by the workflows. This notice is an inventory aid, not a replacement for
the upstream license texts and not legal advice.
