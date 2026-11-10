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

`internal data workflow` may be the development source for an adapter, but published
workflows must not import it or assume its filesystem layout.

The existing K562-essential preparation and split implementation currently
lives under `models/data/`; it will be migrated behind this interface only after
the first vertical slice is protected by artifact-level tests.

