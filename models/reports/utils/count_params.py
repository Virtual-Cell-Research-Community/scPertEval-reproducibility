"""Checkpoint parameter counting - internal helper for reports/*.py.

Lives in reports/utils/ rather than reports/ itself: this is a library module,
not a script - import find_checkpoint()/count_params() from another report
(training_report.py does), don't invoke this file directly.

Parameter count is architecture-determined, not fold-determined (verified:
gears's fold_0/1/2 checkpoints all report the identical 3,378,874 tensor
elements), so find_checkpoint() resolves to exactly one checkpoint per model
(the first fold that has one), not every fold.

Every model here saves a different checkpoint format/filename, but they all
bottom out at a flat dict of tensors one way or another:
  - gears/scgpt:  saved_models/<dataset>/fold_<i>/model.pt - a plain state_dict
  - presage:      saved_models/<dataset>/fold_<i>/*.ckpt   - Lightning checkpoint,
                   state_dict nested under checkpoint["state_dict"]
  - sclambda:     saved_models/<dataset>/fold_<i>/ckpt.pth - dict with a single
                   "Net" key holding the state_dict
  - state:        runs/<dataset>_fold_<i>/checkpoints/best.ckpt - Lightning
                   checkpoint, same shape as presage
  - tabicl:       NOT per-fold-trained - it's a frozen pretrained foundation
                   model (in-context learning only; see tabicl/train_predict.py's
                   own docstring: "TabICL's own checkpoint is pretrained; ICL
                   only"), downloaded once via huggingface_hub to
                   ~/.cache/huggingface/hub/models--jingang--TabICL/ and reused
                   identically across every fold - reported from there instead
                   of anywhere under models/tabicl/.
extract_state_dict() below handles the unwrapping generically rather than
hardcoding each of these shapes.

Counts everything in the state_dict (weights + buffers, e.g. BatchNorm's
running_mean/running_var/num_batches_tracked) under "total", and separately
under "trainable" with those known buffer-name patterns excluded - a heuristic,
not a substitute for the live nn.Module's .parameters()/.buffers() split, but
close enough for a resource-report line, and it means this doesn't need any of
the 6 model-specific pixi envs - plain torch (from the repo's own .venv) is
enough, since a state_dict is just tensors, no custom classes to unpickle.
"""
from pathlib import Path

import torch

MODELS_DIR = Path(__file__).resolve().parent.parent.parent  # reports/utils/ -> reports/ -> models/
BUFFER_NAME_HINTS = ("num_batches_tracked", "running_mean", "running_var")


def find_checkpoint(model, dataset):
    if model == "tabicl":
        hits = sorted(Path.home().glob(".cache/huggingface/hub/models--jingang--TabICL/snapshots/*/*.ckpt"))
        return hits[-1] if hits else None
    patterns = {
        "gears": f"{model}/saved_models/{dataset}/fold_*/model.pt",
        "scgpt": f"{model}/saved_models/{dataset}/fold_*/model.pt",
        "presage": f"{model}/saved_models/{dataset}/fold_*/*.ckpt",
        "sclambda": f"{model}/saved_models/{dataset}/fold_*/ckpt.pth",
        "state": f"{model}/runs/{dataset}_fold_*/checkpoints/best.ckpt",
    }
    hits = sorted(MODELS_DIR.glob(patterns[model]))
    return hits[0] if hits else None


def extract_state_dict(obj):
    if isinstance(obj, dict) and obj and all(torch.is_tensor(v) for v in obj.values()):
        return obj
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "Net", "model"):
            if key in obj:
                return extract_state_dict(obj[key])
    raise ValueError("couldn't find a flat tensor state_dict in this checkpoint")


def count_params(path):
    try:
        obj = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # noqa: BLE001 - retry legacy checkpoints that reject weights_only
        obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = extract_state_dict(obj)
    total = sum(v.numel() for v in sd.values())
    trainable = sum(v.numel() for k, v in sd.items() if not k.endswith(BUFFER_NAME_HINTS))
    return total, trainable
