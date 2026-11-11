"""Compatibility corrections carried by scArchon's trVAE reference container."""

from __future__ import annotations

import torch


def apply_scarchon_patches() -> None:
    """Apply the two material fixes between official trvaep 0.1.0 and scArchon."""
    import trvaep.model._losses as losses
    import trvaep.model.train as training
    from trvaep.model import helper_module
    from trvaep.utils import one_hot_encoder

    def decoder_forward(self, z, c=None):
        if c is not None:
            c = one_hot_encoder(c, n_cls=self.n_classes)
            z = torch.cat((z, c), dim=-1)
        reconstructed = self.FC(z)
        if not self.use_mmd:
            return reconstructed
        # The published 0.1.0 wheel applies ReLU directly to the decoder input here. scArchon
        # restores the intended first linear block before MMD, matching upstream git HEAD.
        first_hidden = self.FC.L0(z)
        if hasattr(self.FC, "B0"):
            first_hidden = self.FC.B0(first_hidden)
        first_hidden = self.FC.A0(first_hidden)
        if hasattr(self.FC, "D0"):
            first_hidden = self.FC.D0(first_hidden)
        return reconstructed, first_hidden

    def corrected_mmd(n_conditions, beta):
        def mmd_loss(values, labels):
            alphas = [
                1e-6,
                1e-5,
                1e-4,
                1e-3,
                1e-2,
                1e-1,
                1,
                5,
                10,
                15,
                20,
                25,
                30,
                35,
                100,
                1e3,
                1e4,
                1e5,
                1e6,
            ]
            groups = losses.partition(values, labels, n_conditions)
            loss = torch.zeros(1, device=values.device)
            for left in range(len(groups)):
                if groups[left].size(0) < 2:
                    continue
                for right in range(left):
                    if groups[right].size(0) < 2:
                        continue
                    statistic = losses.MMDStatistic(groups[left].size(0), groups[right].size(0))
                    loss += statistic(groups[left], groups[right], alphas=alphas)
            return beta * loss

        return mmd_loss

    helper_module.Decoder.forward = decoder_forward
    losses.mmd = corrected_mmd
    # Trainer imported this name directly at module import time.
    training.mmd = corrected_mmd
