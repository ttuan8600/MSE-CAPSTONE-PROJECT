import sys
from pathlib import Path
# ensure root on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.models.eeg_encoder import EEGEncoder, AudioEncoder, MultimodalFusion

print('EEGEncoder output shape', EEGEncoder()(torch.randn(2,28,100)).shape)
print('AudioEncoder output shape', AudioEncoder()(torch.randn(2,13,200)).shape)
fusion = MultimodalFusion()
eeg_feat = torch.randn(2,128)
audio_feat = torch.randn(2,128)
print('Fusion output shape', fusion(eeg_feat,audio_feat).shape)
