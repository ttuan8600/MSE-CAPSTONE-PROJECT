"""Test FACED data loader."""

import sys
sys.path.insert(0, r"c:\Users\ttuan8600\Documents\My Projects\MSE-CAPSTONE-PROJECT")

from src.preprocessing import create_faced_dataloader

# Create FACED dataloader
print("Testing FACED DataLoader...")
faced_dir = r"c:\Users\ttuan8600\Documents\My Projects\MSE-CAPSTONE-PROJECT\data\raw\Processed_data\Processed_data"

dataloader, dataset = create_faced_dataloader(
    data_dir=faced_dir,
    batch_size=8,
    window_size=512,
    shuffle=False,
    subjects=[0, 1, 2],  # Test with first 3 subjects
)

print(f"✓ FACED dataset created with {len(dataset)} windows")

# Get a batch
batch = next(iter(dataloader))
eeg_batch, label_batch = batch

print(f"\nBatch shapes:")
print(f"  EEG: {eeg_batch.shape} (expected: [8, 28, 512])")
print(f"  Labels: {label_batch.shape} (expected: [8])")

assert eeg_batch.shape == (8, 28, 512), f"EEG shape mismatch: {eeg_batch.shape}"
assert label_batch.shape == (8,), f"Label shape mismatch: {label_batch.shape}"

print("\n✅ FACED DataLoader works correctly!")
