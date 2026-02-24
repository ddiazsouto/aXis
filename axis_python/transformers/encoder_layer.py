"""
Complete transformer encoder layer combining attention and feed-forward.
This is one building block - BERT stacks 6 of these for all-MiniLM-L6-v2.
"""

import torch
import torch.nn as nn
from .attention import BertAttention


class BertIntermediate(nn.Module):
    """
    Intermediate layer (first part of feed-forward network).
    This matches BERT's 'intermediate' structure.
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # all-MiniLM-L6-v2 was trained with the tanh-approximate GELU (same as original BERT).
        # PyTorch's default nn.GELU() uses the exact formula, which gives different values.
        self.intermediate_act_fn = nn.GELU(approximate='tanh')
    
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    """
    Output layer (second part of feed-forward network with residual and normalization).
    This matches BERT's 'output' structure.
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class EncoderLayer(nn.Module):
    """
    Complete BERT encoder layer matching official structure:
    - attention (with self-attention and output projection)
    - intermediate (feed-forward expansion)
    - output (feed-forward compression with residual)
    """
    
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
