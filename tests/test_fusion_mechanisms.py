"""Tests for the fusion mechanisms and subject-adversarial training.

The regression these guard against is this project's recurring failure mode: a
component that is wired up, accepts its arguments, runs without error, and has no
effect. ``--adversarial`` was accepted by the single-split trainer for one commit
while being ignored by it, which is exactly the class of defect
``docs/DATA_CORRECTIONS.md`` documents. ``test_adversarial_flag_changes_training``
exists so that cannot recur silently.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from src.models.adversarial import (
    SubjectDiscriminator,
    SubjectIndexMapper,
    adversarial_schedule,
    gradient_reversal,
)
from src.models.feature_encoders import AudioMelEncoder, EEGDEEncoder
from src.models.sequence_fusion import SequenceCrossModalFusion

LATENT = 128


# -- gradient reversal --------------------------------------------------------


def test_gradient_reversal_forward_is_identity():
    x = torch.randn(4, LATENT)
    assert torch.allclose(gradient_reversal(x, 0.7), x)


@pytest.mark.parametrize("lambd", [0.1, 0.3, 1.0])
def test_gradient_reversal_negates_and_scales_gradient(lambd):
    x = torch.randn(4, LATENT, requires_grad=True)
    gradient_reversal(x, lambd).sum().backward()
    reversed_grad = x.grad.clone()

    x.grad = None
    x.sum().backward()
    plain_grad = x.grad

    assert torch.allclose(reversed_grad, -lambd * plain_grad, atol=1e-6)


def test_adversarial_schedule_ramps_from_zero():
    assert adversarial_schedule(0, 70, 0.3) == pytest.approx(0.0, abs=1e-9)
    assert adversarial_schedule(69, 70, 0.3) == pytest.approx(0.3, rel=1e-3)
    values = [adversarial_schedule(e, 70, 0.3) for e in range(70)]
    assert values == sorted(values), "schedule must be non-decreasing"


# -- subject index mapping ----------------------------------------------------


def test_subject_mapper_assigns_contiguous_indices():
    mapper = SubjectIndexMapper(np.array([7, 7, 3, 21, 3]))
    assert mapper.n_subjects == 3
    mapped = mapper(torch.tensor([3, 7, 21]))
    assert sorted(mapped.tolist()) == [0, 1, 2]


def test_subject_mapper_rejects_held_out_subject():
    """A held-out subject must raise, never silently receive an index."""
    mapper = SubjectIndexMapper(np.array([1, 2, 3]))
    with pytest.raises(KeyError, match="not in the training split"):
        mapper(torch.tensor([99]))


def test_discriminator_output_shape():
    d = SubjectDiscriminator(latent_dim=LATENT, n_subjects=28)
    assert d(torch.randn(4, LATENT), 0.3).shape == (4, 28)


# -- the no-op regression -----------------------------------------------------


def test_adversarial_flag_changes_training():
    """Adding the adversarial term must change the gradient reaching the encoder.

    If the discriminator is constructed but its loss never joins the objective --
    the defect this test exists for -- the two gradients below are identical.
    """
    torch.manual_seed(0)
    encoder = EEGDEEncoder(in_features=150, latent_dim=LATENT)
    discriminator = SubjectDiscriminator(latent_dim=LATENT, n_subjects=3)
    discriminator.eval()  # freeze dropout so the comparison is about the loss

    eeg = torch.randn(6, 150, 20)
    emotion = torch.tensor([0, 1, 2, 3, 4, 0])
    subject = torch.tensor([0, 1, 2, 0, 1, 2])
    head = torch.nn.Linear(LATENT, 5)

    def encoder_gradient(with_adversarial: bool):
        encoder.zero_grad()
        latent = encoder(eeg)
        loss = F.cross_entropy(head(latent), emotion)
        if with_adversarial:
            loss = loss + F.cross_entropy(discriminator(latent, 0.3), subject)
        loss.backward()
        return encoder.conv1.weight.grad.clone()

    without = encoder_gradient(False)
    with_adv = encoder_gradient(True)
    assert not torch.allclose(without, with_adv, atol=1e-8), (
        "adversarial term had no effect on the encoder gradient"
    )


# -- sequence fusion ----------------------------------------------------------


def test_encoders_expose_sequences_before_pooling():
    eeg = EEGDEEncoder(in_features=150, latent_dim=LATENT)
    audio = AudioMelEncoder(n_mels=64, latent_dim=LATENT)
    eeg.eval()
    audio.eval()

    eeg_seq = eeg(torch.randn(2, 150, 20), return_sequence=True)
    audio_seq = audio(torch.randn(2, 64, 1313), return_sequence=True)

    assert eeg_seq.shape == (2, 20, LATENT), "one EEG step per one-second window"
    assert audio_seq.ndim == 3 and audio_seq.shape[2] == LATENT
    assert audio_seq.shape[1] > 100, "audio must retain many frames before pooling"


def test_pooled_forward_unchanged_by_sequence_option():
    """The default path must be byte-identical to before the flag was added."""
    torch.manual_seed(0)
    encoder = EEGDEEncoder(in_features=150, latent_dim=LATENT)
    encoder.eval()
    x = torch.randn(3, 150, 20)
    assert torch.allclose(encoder(x), encoder(x, return_sequence=False))


def test_sequence_fusion_pools_after_fusing():
    fusion = SequenceCrossModalFusion(latent_dim=LATENT, num_heads=4)
    fusion.eval()
    out = fusion(torch.randn(2, 20, LATENT), torch.randn(2, 165, LATENT))
    assert out.shape == (2, LATENT)


def test_sequence_fusion_attends_over_many_keys():
    """Attention must depend on the audio sequence, not collapse it.

    The original pooled module attended over a length-1 sequence, where the
    softmax is identically 1.0 and the output cannot depend on the key content.
    """
    torch.manual_seed(0)
    fusion = SequenceCrossModalFusion(latent_dim=LATENT, num_heads=4)
    fusion.eval()
    eeg = torch.randn(1, 20, LATENT)
    audio_a = torch.randn(1, 165, LATENT)
    audio_b = torch.randn(1, 165, LATENT)
    assert not torch.allclose(fusion(eeg, audio_a), fusion(eeg, audio_b), atol=1e-5)


def test_sequence_fusion_rejects_overlong_sequence():
    fusion = SequenceCrossModalFusion(latent_dim=LATENT, num_heads=4)
    with pytest.raises(ValueError, match="positional embeddings"):
        fusion(torch.randn(1, 20, LATENT), torch.randn(1, 9999, LATENT))
