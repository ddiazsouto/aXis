"""
Custom MiniLM Transformer Implementation in PyTorch
Step 1: Environment Setup and Config Loading
Step 2: Embeddings Layer Implementation
"""

import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path



class Embeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    
    def __init__(self, model_name):
        super().__init__()
        relative_models_path = Path(__file__).parent.parent.joinpath('models')
        with open(relative_models_path.joinpath(f'{model_name}/config.json'), 'r') as f:
            config = json.load(f)
        for key, value in config.items():
            setattr(self, key, value)

        self.word_embeddings = nn.Embedding(self.vocab_size, self.hidden_size)
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, self.hidden_size)
        self.token_type_embeddings = nn.Embedding(self.type_vocab_size, self.hidden_size)
        self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps)
        self.dropout = nn.Dropout(self.hidden_dropout_prob)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor = None,
        position_ids: torch.Tensor = None
    ):
        """
        Description:
            Forward pass for embedding layer.
        
        Args:
            input_ids: torch.Tensor 
                of shape (batch_size, sequence_length) containing token IDs
            token_type_ids: torch.Tensor 
                of shape (batch_size, sequence_length), optional. 
                Defaults to zeros if not provided.
            position_ids: torch.Tensor 
                of shape (batch_size, sequence_length), optional.
                Defaults to sequential positions if not provided.
        
        Returns:
            embeddings: torch.Tensor
                Shape (batch_size, sequence_length, hidden_size)
        """
        seq_length = input_ids.size(1)
        device = input_ids.device
        
        if position_ids is None:
            position_ids = torch.arange(seq_length, device=device).unsqueeze(0).expand_as(input_ids)
        
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids, device=device)
        
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
                
        return self.LayerNorm(
            word_embeddings + position_embeddings + token_type_embeddings
        )        
