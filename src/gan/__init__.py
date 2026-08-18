"""Placeholder for generative augmentation. Intentionally empty.

The original project proposal listed a GAN for synthetic EEG augmentation as
research objective RO1. That objective was **not pursued**, and this package was
never implemented. It is retained only so that references to it in the proposal
resolve to an explicit statement rather than a missing module.

The decision and its rationale are recorded in the report methodology chapter
("Scope Decision: Generative Augmentation Not Pursued"). In short: sample volume
is not the binding constraint on this problem -- the corrected pipeline yields
8,400 labelled EEG trials from 42 subjects -- and a generator trained on the
training subjects cannot synthesise the inter-subject variability that dominates
the subject-independent error. Generative augmentation evaluated against a
subject-independent baseline remains open future work.
"""
