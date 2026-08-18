"""Subject-adversarial training, to attack the measured cross-subject failure.

Finding 4 of this project localises the bottleneck: the EEG encoder reaches
99.75% training accuracy while transferring at 45.90%, so its representation
varies more by individual than by emotion. Euclidean alignment addresses part of
this at the *feature* level; this module addresses it at the *representation*
level.

A gradient reversal layer (Ganin & Lempitsky, 2015) is placed between the EEG
encoder and an auxiliary classifier that predicts *which subject* a trial came
from. During the backward pass the gradient flowing into the encoder is negated,
so the encoder is trained to make the subject **un**identifiable while the
auxiliary head simultaneously tries to identify it. At convergence the latent
retains what predicts emotion and discards what predicts identity.

The adversarial weight is ramped from zero rather than applied from the first
step: early in training the subject head is untrained, so its gradients are
noise, and reversing noise destabilises the encoder.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class GradientReversal(torch.autograd.Function):
    """Identity forwards; negated and scaled gradient backwards."""

    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def gradient_reversal(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


class SubjectDiscriminator(nn.Module):
    """Predicts the subject id from a latent, through a gradient reversal layer.

    Deliberately small. A high-capacity discriminator wins outright, driving the
    encoder to destroy its latent to escape; the goal is a useful adversary, not
    a strong one.
    """

    def __init__(self, latent_dim: int = 128, n_subjects: int = 42, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_subjects),
        )

    def forward(self, latent: torch.Tensor, lambd: float) -> torch.Tensor:
        return self.net(gradient_reversal(latent, lambd))


def adversarial_schedule(epoch: int, total_epochs: int, max_lambda: float = 0.3) -> float:
    """Ramp lambda from 0 to ``max_lambda`` on the DANN schedule.

    ``2 / (1 + exp(-10p)) - 1`` for progress ``p``, which is near zero for the
    first few epochs and saturates around three-quarters of the way through.
    """
    if total_epochs <= 1:
        return max_lambda
    progress = epoch / (total_epochs - 1)
    return float(max_lambda * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0))


class SubjectIndexMapper:
    """Maps arbitrary subject ids onto contiguous indices for the discriminator.

    Only subjects present in the *training* split may be mapped: the held-out
    subjects must never acquire an index, since that would imply the adversary
    had seen them.
    """

    def __init__(self, train_subject_ids: np.ndarray):
        unique = np.unique(train_subject_ids)
        self._index = {int(s): i for i, s in enumerate(unique)}
        self.n_subjects = len(unique)

    def __call__(self, subject_ids: torch.Tensor) -> torch.Tensor:
        try:
            mapped = [self._index[int(s)] for s in subject_ids]
        except KeyError as exc:
            raise KeyError(
                f"subject {exc.args[0]} is not in the training split; the "
                f"subject discriminator must not be shown held-out subjects"
            ) from exc
        return torch.tensor(mapped, dtype=torch.long, device=subject_ids.device)
