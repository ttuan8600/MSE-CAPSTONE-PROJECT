"""EEG Encoder models for emotion recognition.

Implements CNN-based and CNN-LSTM hybrid architectures for encoding EEG signals.
Used for pre-training on FACED dataset and fine-tuning on EAV multimodal data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGEncoder(nn.Module):
    """CNN-based EEG encoder for feature extraction.
    
    Processes raw EEG signals (28 channels, variable time steps) through
    temporal convolutions to extract discriminative emotion features.
    
    Architecture:
    - Input: (batch_size, 28, time_steps)
    - Conv blocks with temporal kernels
    - Adaptive pooling to fixed dimension
    - Output: (batch_size, latent_dim)
    """
    
    def __init__(self, in_channels=28, latent_dim=128):
        """Initialize EEG encoder.
        
        Parameters
        ----------
        in_channels : int, default=28
            Number of EEG channels (standard).
        latent_dim : int, default=128
            Dimension of latent representation.
        """
        super().__init__()
        self.in_channels = in_channels
        self.latent_dim = latent_dim
        
        # Temporal convolution blocks
        # Block 1: (28, T) -> (64, T/2)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        # Block 2: (64, T/2) -> (128, T/4)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # Block 3: (128, T/4) -> (256, T/8)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Block 4: (256, T/8) -> (256, T/16)
        self.conv4 = nn.Conv1d(256, 256, kernel_size=5, stride=2, padding=2)
        self.bn4 = nn.BatchNorm1d(256)
        
        # Adaptive pooling to fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # Projection to latent dimension
        self.fc = nn.Linear(256, latent_dim)
        
    def forward(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input EEG signal of shape (batch_size, 28, time_steps).
            
        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        """
        # Conv block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Conv block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        # Adaptive pooling to (batch_size, 256, 1)
        x = self.adaptive_pool(x)
        x = x.squeeze(-1)  # (batch_size, 256)
        
        # Project to latent space
        x = self.fc(x)
        
        return x


class EEGEncoderLSTM(nn.Module):
    """CNN-LSTM hybrid EEG encoder for enhanced temporal modeling.
    
    Combines convolutional feature extraction with LSTM for better
    temporal dependency capture across EEG channels.
    
    Architecture:
    - Input: (batch_size, 28, time_steps)
    - Conv layers for feature extraction
    - LSTM for temporal modeling
    - Output: (batch_size, latent_dim)
    """
    
    def __init__(self, in_channels=28, hidden_dim=64, latent_dim=128, num_layers=2):
        """Initialize EEG encoder with LSTM.
        
        Parameters
        ----------
        in_channels : int, default=28
            Number of EEG channels.
        hidden_dim : int, default=64
            LSTM hidden dimension.
        latent_dim : int, default=128
            Dimension of latent representation.
        num_layers : int, default=2
            Number of LSTM layers.
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # CNN feature extractor (reduce temporal dimension)
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        
        # LSTM temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Projection to latent space
        self.fc = nn.Linear(hidden_dim * 2, latent_dim)
        
    def forward(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input EEG signal of shape (batch_size, 28, time_steps).
            
        Returns
        -------
        torch.Tensor
            Latent representation of shape (batch_size, latent_dim).
        """
        # CNN feature extraction
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Prepare for LSTM: (batch_size, time_steps, features)
        x = x.transpose(1, 2)
        
        # LSTM processing
        _, (h_n, _) = self.lstm(x)
        
        # Use final hidden state from both directions
        # h_n shape: (num_layers * num_directions, batch_size, hidden_dim)
        # For bidirectional LSTM: (num_layers*2, batch_size, hidden_dim)
        h_n = h_n.transpose(0, 1)  # (batch_size, num_layers*2, hidden_dim)
        h_n = h_n[:, -2:, :].reshape(h_n.size(0), -1)  # Last layer both directions
        
        # Project to latent space
        x = self.fc(h_n)
        
        return x


class EmotionClassifier(nn.Module):
    """Emotion classifier head for pre-training and fine-tuning.
    
    Takes EEG encoder output and classifies emotion.
    Used for supervised pre-training on FACED dataset.
    """
    
    def __init__(self, latent_dim=128, num_emotions=5):
        """Initialize emotion classifier.
        
        Parameters
        ----------
        latent_dim : int, default=128
            Dimension of input features from encoder.
        num_emotions : int, default=5
            Number of emotion classes.
        """
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_emotions)
        )
    
    def forward(self, x):
        """Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Latent features from encoder of shape (batch_size, latent_dim).
            
        Returns
        -------
        torch.Tensor
            Logits of shape (batch_size, num_emotions).
        """
        return self.fc(x)
