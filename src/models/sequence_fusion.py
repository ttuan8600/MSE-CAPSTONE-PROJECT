"""Cross-modal attention over *sequences*, fused before temporal pooling.

Why this module exists
----------------------
:class:`~src.models.attention_fusion.CrossModalAttentionFusion` receives two
already-pooled vectors and calls ``unsqueeze(1)`` on each, so its multi-head
attention operates on sequences of length one. A softmax over a single key is
identically 1.0 regardless of the query, which means **no attention is
computed**: the module reduces to a gated linear combination of two vectors,
with the attention projections acting as ordinary linear layers.

That is not a bug in the sense of raising an error --- it trains and it
converges --- but it does mean the project's headline architectural claim was
never actually tested. Both encoders pool globally over time before fusing, so
the mechanism cannot align a vocal event with a concurrent neural response even
in principle.

This module tests the claim properly. Each encoder exposes its pre-pooling
sequence, cross-attention runs over those sequences, and pooling happens after
fusion:

* EEG band power  : 20 one-second windows -> ``(B, 20, D)``
* Audio log-mel   : ~165 frames after three stride-2 blocks -> ``(B, 165, D)``

An EEG window can therefore attend to the specific audio frames that overlap it,
which is the operation the thesis proposed and did not implement.
"""

from __future__ import annotations

import torch
import torch.nn as nn

#: Sequence lengths are fixed by the preprocessing cache, so learned positional
#: embeddings are allocated for a generous maximum and sliced.
MAX_EEG_STEPS = 64
MAX_AUDIO_STEPS = 256


class SequenceCrossModalFusion(nn.Module):
    """Bidirectional cross-attention between two temporal sequences.

    Parameters
    ----------
    latent_dim, num_heads, dropout:
        As for the pooled variant, so parameter counts stay comparable.

    Notes
    -----
    Positional embeddings are learned per stream. Without them attention is
    permutation-invariant over time, which would defeat the purpose --- temporal
    correspondence between the streams is the entire hypothesis.
    """

    def __init__(self, latent_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")

        self.latent_dim = latent_dim
        self.eeg_position = nn.Parameter(torch.zeros(1, MAX_EEG_STEPS, latent_dim))
        self.audio_position = nn.Parameter(torch.zeros(1, MAX_AUDIO_STEPS, latent_dim))
        nn.init.trunc_normal_(self.eeg_position, std=0.02)
        nn.init.trunc_normal_(self.audio_position, std=0.02)

        self.eeg_to_audio = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.audio_to_eeg = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm_eeg = nn.LayerNorm(latent_dim)
        self.norm_audio = nn.LayerNorm(latent_dim)
        self.norm_out = nn.LayerNorm(latent_dim)

        self.project = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

    def _add_position(self, x: torch.Tensor, table: nn.Parameter) -> torch.Tensor:
        steps = x.size(1)
        if steps > table.size(1):
            raise ValueError(
                f"sequence of {steps} steps exceeds the {table.size(1)} positional "
                f"embeddings allocated; raise MAX_*_STEPS"
            )
        return x + table[:, :steps]

    def forward(self, eeg_seq: torch.Tensor, audio_seq: torch.Tensor) -> torch.Tensor:
        """``(B, T_eeg, D)`` and ``(B, T_audio, D)`` -> ``(B, D)``."""
        eeg_seq = self._add_position(eeg_seq, self.eeg_position)
        audio_seq = self._add_position(audio_seq, self.audio_position)

        # Each EEG window attends over all audio frames, and vice versa. Unlike
        # the pooled variant these softmaxes are over many keys, so they can
        # actually select.
        eeg_attended, _ = self.eeg_to_audio(
            query=eeg_seq, key=audio_seq, value=audio_seq
        )
        eeg_attended = self.norm_eeg(eeg_attended + eeg_seq)

        audio_attended, _ = self.audio_to_eeg(
            query=audio_seq, key=eeg_seq, value=eeg_seq
        )
        audio_attended = self.norm_audio(audio_attended + audio_seq)

        # Pool only now, after the streams have seen each other.
        pooled = torch.cat(
            [eeg_attended.mean(dim=1), audio_attended.mean(dim=1)], dim=1
        )
        return self.norm_out(self.project(pooled))
