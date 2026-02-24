"""
Feed-forward network for BERT transformer layers.
Simple two-layer MLP with GELU activation.
"""

import torch.nn as nn


class FeedForward(nn.Module):
    """
    Uses the intermediate size parameter, then activate it with GELU, and finally project back to the hidden size.
    Refines attention oputput and deals with the vanishing gradient problem (GELU).
    """
    def __init__(self, config):
        super().__init__()
        self.dense1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.dense2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states):
        x = self.dense1(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.dense2(x)
        return x