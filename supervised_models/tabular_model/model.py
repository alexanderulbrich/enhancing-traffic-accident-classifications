import torch
import torch.nn as nn

class ResNetBlock(nn.Module):
    def __init__(self, hidden_dim, dropout):
        super().__init__()
        self.block = nn.Sequential(
            # First part: BN -> Linear -> ReLU -> Dropout
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            # Second part: Linear -> Dropout
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Residual connection: add input x to the transformed output.
        return x + self.block(x)

class TabularModelNN(nn.Module):
    def __init__(self, num_classes, tabular_dim, hidden_dim, num_resnet_blocks, dropout):
        """
        num_classes: number of output classes.
        tabular_dim: dimension of the tabular features.
        hidden_dim: dimension of the hidden layers.
        num_resnet_blocks: number of ResNet blocks to use.
        dropout: dropout probability.
        """
        super().__init__()
        
        # Process the tabular features with two MLP layers.
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_dim, tabular_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(tabular_dim * 2, tabular_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # The linear layer projects the concatenated features (processed tabular + text)
        # into the common hidden dimension.
        self.upscale_linear_layer = nn.Linear(tabular_dim*2, hidden_dim)
        
        # Define a stack of ResNet blocks.
        self.blocks = nn.Sequential(
            *[ResNetBlock(hidden_dim, dropout) for _ in range(num_resnet_blocks)]
        )
        
        # The prediction head.
        self.prediction = nn.Sequential(
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, hidden_dim // 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 32, num_classes)
        )
        
    def forward(self, tabular_embedding):
        """
        tabular_embedding: tensor of shape (batch_size, tabular_dim)
        """
        # Process the tabular features.
        processed_tabular = self.tabular_mlp(tabular_embedding)
        
        # Project the concatenated features into the hidden space.
        x = self.upscale_linear_layer(processed_tabular)
        
        # Apply the series of residual blocks.
        x = self.blocks(x)
        
        # Get the final prediction logits.
        logits = self.prediction(x)
        return logits
