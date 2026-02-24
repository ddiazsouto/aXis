"""
BERT encoder that stacks multiple transformer layers.
For all-MiniLM-L6-v2, this contains 6 identical encoder layers.
"""

import torch
import torch.nn as nn
from .encoder_layer import EncoderLayer


class BertEncoder(nn.Module):
    """
    Description:
        Stack of transformer encoder layers.
        all-MiniLM-L6-v2 uses 6 layers (hence the "L6" in the name).
        
    Args:
        config: Configuration object with:
            - num_hidden_layers: Number of encoder layers to stack (6 for all-MiniLM-L6-v2)
            - All other parameters needed by EncoderLayer
    """
    
    def __init__(self, config):
        super().__init__()

        self.layer = nn.ModuleList([
            EncoderLayer(config) 
            for _ in range(config.num_hidden_layers)
        ])
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Pass input through all encoder layers sequentially.
        
        Args:
            hidden_states: [batch_size, seq_length, hidden_size]
                          Initial embeddings from the Embeddings layer
            attention_mask: [batch_size, 1, 1, seq_length] (optional)
                           Mask to prevent attention to padding tokens
        
        Returns:
            output: [batch_size, seq_length, hidden_size]
                   Final hidden states after all 6 layers
        """
        for encoder_layer in self.layer:
            hidden_states = encoder_layer(hidden_states, attention_mask)

        return hidden_states
