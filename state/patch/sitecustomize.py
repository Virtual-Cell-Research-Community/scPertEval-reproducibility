"""Adds early stopping to STATE's training loop via a Python startup hook, not a fork of the
installed package.

`state tx train` runs as a subprocess (see train_predict.py) rather than being imported
in-process, so a plain monkeypatch from train_predict.py itself would never reach it —
sitecustomize.py is Python's own mechanism for this instead: any module with that exact name
found on sys.path gets auto-imported at interpreter startup, before the subprocess's own code
runs, as long as its directory is on PYTHONPATH. train_predict.py adds this directory to
PYTHONPATH only for the `state tx train` subprocess call (not `state tx predict`), so this has
no effect on bare/manual `state` CLI invocations outside train_predict.py, and no effect on
predict.

state's own get_checkpoint_callbacks() (state/tx/utils/__init__.py) builds only a
ModelCheckpoint(monitor="val_loss", ...) — no EarlyStopping callback exists anywhere in the
installed state version (arc-state==0.11.1), confirmed directly against
state/_cli/_tx/_train.py's run_tx_train(). That function imports get_checkpoint_callbacks
*locally* (`from ...tx.utils import get_checkpoint_callbacks`, inside the function body, not
at module load time), so replacing the state.tx.utils module attribute here — before
run_tx_train() ever executes — is picked up transparently: Python resolves that import against
whatever's currently in sys.modules at call time, which by then is this patched version.

patience is read from an environment variable rather than hardcoded, since sitecustomize.py is
auto-imported with no way to receive arguments directly — train_predict.py sets it alongside
PYTHONPATH on the same subprocess call. min_delta is left at Lightning's own default (0.0, i.e.
any strict improvement counts and resets the wait counter) rather than given a value here: a
value of 0.01 was tried first and simulated directly against a real measured val_loss trend
(cell_set_len=64 full-scale diagnostic, see train_predict.py's git history) — it produced the
exact same stop point as 0.0 at every patience tested, because the real step-to-step "new low"
moves in that data are almost always bigger than 0.01, even amid noise. It wasn't filtering
anything, just adding an unearned knob, so it was dropped rather than kept at a value doing
nothing.
"""

import os

from lightning.pytorch.callbacks import EarlyStopping

import state.tx.utils as _utils

_original_get_checkpoint_callbacks = _utils.get_checkpoint_callbacks


def _get_checkpoint_callbacks_with_early_stopping(output_dir, name, val_freq, _ckpt_every_n_steps):
    callbacks = _original_get_checkpoint_callbacks(output_dir, name, val_freq, _ckpt_every_n_steps)
    patience = int(os.environ.get("STATE_EARLY_STOP_PATIENCE", 10))
    # verbose=True (not EarlyStopping's own default of False) so that if it actually
    # triggers, the reason ("did not improve in the last N records... Signaling Trainer to
    # stop") is printed to stdout — otherwise a run stopped by EarlyStopping is silently
    # indistinguishable from one that reached max_steps normally. Matches PRESAGE's own
    # early_stop_callback, which also sets verbose=True.
    callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=patience, verbose=True))
    return callbacks


_utils.get_checkpoint_callbacks = _get_checkpoint_callbacks_with_early_stopping
