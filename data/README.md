# Data workflows

This directory owns public raw-data acquisition, verification, preparation, and
split generation. Raw and processed artifacts are ignored by Git.

Each supported dataset will have:

1. a declaration under `configs/datasets/`;
2. authoritative download instructions and immutable input checksums;
3. a raw-to-analysis adapter;
4. output validation and a dataset manifest;
5. explicit split files containing condition identifiers; and
6. a small smoke fixture or metadata-only test where licensing permits.

Internal workflows may inform an adapter, but published workflows must be
self-contained and must not assume a private filesystem layout.

The existing K562-essential preparation and split implementation currently
lives under `models/data/`; it will be migrated behind this interface only after
the first vertical slice is protected by artifact-level tests.

