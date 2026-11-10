# Data workflows

This directory owns public raw-data acquisition, verification, preparation, and
split generation. Raw and processed artifacts are ignored by Git.

Each supported dataset has:

1. a declaration under `configs/datasets/`;
2. authoritative download instructions and immutable input checksums;
3. a raw-to-analysis adapter;
4. output validation and a dataset manifest;
5. explicit split files containing condition identifiers; and
6. a small smoke fixture or metadata-only test where licensing permits.

Internal workflows may inform an adapter, but published workflows must be
self-contained and must not assume a private filesystem layout.

## Commands

The root environment is managed by `uv`. Generated data live under `data/` by
default, or under `$SCPERTEVAL_DATA_ROOT` on shared storage.

```bash
uv sync --dev
uv run python -m workflows.data download replogle-weissman-2022-k562-essential
uv run python -m workflows.data prepare replogle-weissman-2022-k562-essential
uv run python -m workflows.data validate replogle-weissman-2022-k562-essential
```

`all` performs all three steps. Existing processed outputs are never
overwritten unless `--force` is passed explicitly. Validation checks the public
data contract and the exact checksum of the previously validated reference
artifact.

The public workflow implements the canonical raw-count transformations directly;
it has no private imports or filesystem assumptions. See
`docs/data-provenance.md` for the input and reference-output contract.

