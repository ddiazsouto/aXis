import torch
import torch.nn as nn
import torch.nn.functional as F


def mean_pooling(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Compute a single sentence embedding by averaging all real (non-padding) token vectors,
    then L2-normalise the result.

    This is the exact pooling strategy used by sentence-transformers for all-MiniLM-L6-v2.

    Args:
        hidden_states:  [batch, seq_len, hidden_size]  – encoder output for every token
        attention_mask: [batch, seq_len]               – 1 for real token, 0 for padding

    Returns:
        pooled: [batch, hidden_size]  – L2-normalised mean embedding
    """    
    mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
    sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
    token_count = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
    mean_embeddings = sum_embeddings / token_count
    return F.normalize(mean_embeddings, p=2, dim=1)


class BertPooler(nn.Module):
    """
    Takes the hidden state of the first ([CLS]) token and projects it through
    a linear layer + tanh activation.

    In standard BERT this is used for classification tasks.
    For sentence-transformers / all-MiniLM-L6-v2, mean_pooling is used instead,
    but the weights (pooler.dense.weight / pooler.dense.bias) are still present
    in the official checkpoint, so we keep this submodule to absorb them cleanly.

    Submodule name MUST be ``pooler`` in the parent model so that
    ``load_state_dict`` maps  ``pooler.dense.*``  correctly.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

        cls_token = hidden_states[:, 0]
        return self.activation(self.dense(cls_token))

