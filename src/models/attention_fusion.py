"""
Enhanced Cross-Modal Attention Fusion for multimodal emotion recognition.
Implements multi-head cross-attention to learn which parts of each modality
are important for distinguishing emotions.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion for EEG and audio streams.
    
    Uses multi-head attention to learn relationships between EEG and audio features.
    Each emotion may attend to different aspects of each modality.
    """
    
    def __init__(self, latent_dim=128, num_heads=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        
        assert latent_dim % num_heads == 0, "latent_dim must be divisible by num_heads"
        
        # EEG -> Audio attention (what audio information is relevant for EEG)
        self.eeg_to_audio_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Audio -> EEG attention (what EEG information is relevant for audio)
        self.audio_to_eeg_attn = nn.MultiheadAttention(
            embed_dim=latent_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization for stability
        self.norm1_eeg = nn.LayerNorm(latent_dim)
        self.norm2_audio = nn.LayerNorm(latent_dim)
        self.norm3_fused = nn.LayerNorm(latent_dim)
        
        # Fusion gate (learn how to combine attended features)
        self.fusion_gate = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid()
        )
        
        # Final projection
        self.final_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim)
        )
    
    def forward(self, eeg_feat, audio_feat):
        """
        Args:
            eeg_feat: (batch_size, latent_dim) EEG features
            audio_feat: (batch_size, latent_dim) Audio features
        
        Returns:
            fused: (batch_size, latent_dim) Fused representation
        """
        # Reshape for multi-head attention (expects seq_len dimension)
        # Treat each sample's features as a sequence of 1 element
        eeg_input = eeg_feat.unsqueeze(1)  # (B, 1, D)
        audio_input = audio_feat.unsqueeze(1)  # (B, 1, D)
        
        # Cross-attention: EEG attends to audio
        # EEG is query, audio is key and value
        eeg_attended, eeg_attn_weights = self.eeg_to_audio_attn(
            query=eeg_input,
            key=audio_input,
            value=audio_input
        )
        eeg_attended = eeg_attended.squeeze(1)  # (B, D)
        eeg_attended = self.norm1_eeg(eeg_attended + eeg_feat)  # Residual + norm
        
        # Cross-attention: Audio attends to EEG
        # Audio is query, EEG is key and value
        audio_attended, audio_attn_weights = self.audio_to_eeg_attn(
            query=audio_input,
            key=eeg_input,
            value=eeg_input
        )
        audio_attended = audio_attended.squeeze(1)  # (B, D)
        audio_attended = self.norm2_audio(audio_attended + audio_feat)  # Residual + norm
        
        # Fusion gate: learn how to weight the attended features
        combined = torch.cat([eeg_attended, audio_attended], dim=1)  # (B, 2D)
        gate = self.fusion_gate(combined)  # (B, D)
        
        # Weight attended features by gate
        gated_eeg = eeg_attended * gate
        gated_audio = audio_attended * (1 - gate)
        
        # Combine
        fused = torch.cat([gated_eeg, gated_audio], dim=1)  # (B, 2D)
        fused = self.final_proj(fused)  # (B, D)
        fused = self.norm3_fused(fused)
        
        return fused


class AttentionFusionNetwork(nn.Module):
    """Complete network with cross-modal attention fusion."""
    
    def __init__(self, eeg_encoder, audio_encoder, classifier):
        """
        Args:
            eeg_encoder: Trained EEG encoder module
            audio_encoder: Trained audio encoder module
            classifier: Emotion classifier module
        """
        super().__init__()
        self.eeg_encoder = eeg_encoder
        self.audio_encoder = audio_encoder
        self.attention_fusion = CrossModalAttentionFusion(latent_dim=128, num_heads=4)
        self.classifier = classifier
    
    def forward(self, eeg, audio):
        eeg_feat = self.eeg_encoder(eeg)
        audio_feat = self.audio_encoder(audio)
        fused = self.attention_fusion(eeg_feat, audio_feat)
        logits = self.classifier(fused)
        return logits


if __name__ == '__main__':
    # Note: Run from project root with: python -m src.models.attention_fusion
    print("To test, run: python -m src.models.attention_fusion from project root")
    print("Cross-Modal Attention Fusion module loaded successfully!")
