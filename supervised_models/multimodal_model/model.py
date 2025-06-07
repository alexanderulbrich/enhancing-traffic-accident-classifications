import torch
import torch.nn as nn
from transformers import XLMRobertaModel

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

class MultiModalNN(nn.Module):
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
        
        # Load the pretrained XLM-RoBERTa model.
        # Its parameters will be updated during training.
        self.text_encoder = XLMRobertaModel.from_pretrained("xlm-roberta-large")
        # Get the text embedding dimension from the pretrained model config.
        text_embedding_dim = self.text_encoder.config.hidden_size
        
        # The linear layer projects the concatenated features (processed tabular + text)
        # into the common hidden dimension.
        self.concat_to_hidden_layer = nn.Linear(tabular_dim * 2 + text_embedding_dim, hidden_dim)
        
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
        
    def forward(self, tabular_embedding, text_input):
        """
        tabular_embedding: tensor of shape (batch_size, tabular_dim)
        text_input: dictionary of inputs for XLM-RoBERTa (e.g., 
                    {'input_ids': ..., 'attention_mask': ...})
                    
        The `text_input` dictionary is expected to be pre-tokenized using the corresponding 
        tokenizer from Hugging Face (e.g., XLMRobertaTokenizer).
        """
        # Process the tabular features.
        processed_tabular = self.tabular_mlp(tabular_embedding)
        
        # Obtain text embeddings from the pretrained XLM-RoBERTa model.
        # The model returns a BaseModelOutputWithPoolingAndCrossAttentions.
        text_outputs = self.text_encoder(**text_input)
        # Apply mean pooling
        text_embedding = self.mean_pooling(
            text_outputs.last_hidden_state,
            text_input['attention_mask']
        )
        
        # Combine the processed tabular features with the text embedding.
        x = torch.cat([processed_tabular, text_embedding], dim=1)
        
        # Project the concatenated features into the hidden space.
        x = self.concat_to_hidden_layer(x)
        
        # Apply the series of residual blocks.
        x = self.blocks(x)
        
        # Get the final prediction logits.
        logits = self.prediction(x)
        return logits
    
    def mean_pooling(self, token_embeddings, attention_mask):
        """
        Perform mean pooling on the token embeddings using the attention mask
        to properly handle padding tokens.
        
        Args:
            token_embeddings: Tensor of shape (batch_size, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch_size, seq_len)
            
        Returns:
            Tensor of shape (batch_size, hidden_size)
        """
        # Convert attention mask to float and unsqueeze last dim
        input_mask_expanded = attention_mask.unsqueeze(-1).float()
        
        # Sum the embeddings along seq_len dimension with mask applied
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        
        # Sum the mask to get the actual sequence lengths
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Divide by the sequence lengths to get mean
        return sum_embeddings / sum_mask    