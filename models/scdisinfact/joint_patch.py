"""Numerical guard required when scDisInFact models many conditions jointly."""

from __future__ import annotations

from collections.abc import Iterable


def choose_present_reference(observed: Iterable[int], *, requested: int | None) -> int | None:
    """Choose an MMD reference represented in the current mini-batch.

    Upstream scDisInFact assumes its globally chosen reference batch is present for every
    condition in every mini-batch. That holds for its small binary examples, but not for a
    joint model with dozens or hundreds of perturbations. With fewer than two represented
    batches there is no cross-batch discrepancy to estimate.
    """
    present = sorted(set(observed))
    if len(present) < 2:
        return None
    if requested in present:
        return requested
    return present[0]


def apply_joint_condition_mmd_patch() -> None:
    """Prevent empty-reference MMD means without changing estimable MMD terms."""
    import torch
    from scDisInFact import loss_function

    original = loss_function.maximum_mean_discrepancy
    if getattr(original, "_scperteval_joint_condition_safe", False):
        return

    def safe_maximum_mean_discrepancy(xs, batch_ids, device, ref_batch=None):
        observed_tensors = torch.unique(batch_ids, sorted=True)
        observed = [int(value.item()) for value in observed_tensors]
        requested = None if ref_batch is None else int(ref_batch.item())
        effective = choose_present_reference(observed, requested=requested)
        if effective is None:
            return xs.sum() * 0.0
        effective_tensor = torch.as_tensor(effective, device=batch_ids.device, dtype=batch_ids.dtype)
        return original(xs=xs, batch_ids=batch_ids, device=device, ref_batch=effective_tensor)

    safe_maximum_mean_discrepancy._scperteval_joint_condition_safe = True
    loss_function.maximum_mean_discrepancy = safe_maximum_mean_discrepancy
