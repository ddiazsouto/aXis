"""
Multi-head attention module for BERT-like transformer encoders.
Matches the all-MiniLM-L6-v2 model architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Description:
        Multi-head self-attention layer for BERT-like transformers.
    
    Args:
        config: Configuration object with attributes:
            - hidden_size: Dimension of hidden states (default: 384 for all-MiniLM-L6-v2)
            - num_attention_heads: Number of attention heads (default: 12)
            - attention_probs_dropout_prob: Dropout probability for attention weights
    """
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_probs_dropout_prob = config.attention_probs_dropout_prob
        
        # Calculate head size
        self.head_size = self.hidden_size // self.num_attention_heads
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
        
        # Linear projections for Q, K, V
        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Output projection
        self.dense = nn.Linear(self.hidden_size, self.hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(self.attention_probs_dropout_prob)
    
    def forward(self, hidden_states, attention_mask=None):
        """
        Apply multi-head self-attention.
        
        Args:
            hidden_states: [batch_size, seq_length, hidden_size]
            attention_mask: [batch_size, 1, 1, seq_length] (optional)
                           Values of 1 for tokens to attend to, 0 for tokens to mask
        
        Returns:
            output: [batch_size, seq_length, hidden_size]
        """
        batch_size, seq_length, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.query(hidden_states)  # [batch, seq, hidden]
        k = self.key(hidden_states)    # [batch, seq, hidden]
        v = self.value(hidden_states)  # [batch, seq, hidden]

        q = q.view(batch_size, seq_length, self.num_attention_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, seq_length, self.num_attention_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, seq_length, self.num_attention_heads, self.head_size).transpose(1, 2)
        
        # Compute attention scores: Q @ K^T / sqrt(head_size)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        # scores shape: [batch, heads, seq, seq]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask: [batch, 1, 1, seq]
            # Convert: where mask==0, set scores to -1e9
            scores = scores + (attention_mask == 0).float() * -1e9
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)  # [batch, heads, seq, head_size]
        
        # Reshape back: [batch, heads, seq, head_size] -> [batch, seq, heads, head_size] -> [batch, seq, hidden]
        context = context.transpose(1, 2).contiguous()
        context = context.view(batch_size, seq_length, self.hidden_size)
        
        # Final output projection
        output = self.dense(context)
        
        return output
