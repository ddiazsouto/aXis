"""
Multi-head attention module for BERT-like transformer encoders.
Matches the all-MiniLM-L6-v2 model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BertSelfAttention(nn.Module):
    """
    Multi-head self-attention mechanism (Q, K, V projections and attention computation).
    This matches BERT's 'attention.self' submodule.
    """
    
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = config.hidden_size
        
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(self, hidden_states, attention_mask=None):
        q = self.transpose_for_scores(self.query(hidden_states))
        k = self.transpose_for_scores(self.key(hidden_states))
        v = self.transpose_for_scores(self.value(hidden_states))
        
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.attention_head_size)
        
        # attention_mask is already an additive bias: 0.0 for real tokens, -1e9 for padding.
        # Simply add it directly to the logits before softmax.
        if attention_mask is not None:
            scores = scores + attention_mask
        
        attention_probs = F.softmax(scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        context = torch.matmul(attention_probs, v)
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.size()[:-2] + (self.all_head_size,)
        context = context.view(new_context_shape)
        
        return context


class BertSelfOutput(nn.Module):
    """
    Output projection and layer normalization for self-attention.
    This matches BERT's 'attention.output' submodule.
    """
    
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    """
    Complete attention module combining self-attention and output projection.
    This matches BERT's 'attention' structure with 'self' and 'output' submodules.
    """
    
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
    
    def forward(self, hidden_states, attention_mask=None):
        self_output = self.self(hidden_states, attention_mask)
        attention_output = self.output(self_output, hidden_states)
        return attention_output


# Keep backward compatibility
MultiHeadAttention = BertAttention
