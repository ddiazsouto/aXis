from pathlib import Path
from typing import List, Union

import json
import numpy as np
import torch
import torch.nn as nn

from .embeddings import Embeddings
from .bert_encoder import BertEncoder
from .pooling import mean_pooling, BertPooler
from axis_python.tokeniser.bert_tokenizer import BertTokenizer


class BertModel(nn.Module):
    """
    Complete BERT model built from scratch.

    Architecture:
        1. Embeddings   token + position + token_type + LayerNorm + dropout
        2. BertEncoder  6 stacked transformer layers (attention + feed-forward)
        3. mean_pooling  produces a single normalised sentence vector

    Args:
        model_path: Path to the model directory inside axis_python/models/  (e.g. 'all-MiniLM-L6-v2')
    """

    def __init__(self, model_path: Path, load_weights: bool = True):
        super().__init__()
        model_path = Path(model_path)
        self.embeddings = Embeddings(model_path)
        self.encoder    = BertEncoder(self.embeddings)
        self.pooler     = BertPooler(self.embeddings.hidden_size)
        self.model_path = model_path
        self.tokenizer  = BertTokenizer(model_path)

        if load_weights:
            weights_dir = model_path / 'weights'
            if not weights_dir.exists():
                raise FileNotFoundError(f"No weights directory found at {weights_dir}")
            raw: dict = {}
            for shard_path in sorted(weights_dir.glob("weights_*.json")):
                with open(shard_path) as f:
                    raw.update(json.load(f))

            chunks: dict = {}
            state_dict: dict = {}
            for k, v in raw.items():
                if "__chunk_" in k:
                    base, _ = k.rsplit("__chunk_", 1)
                    chunks.setdefault(base, []).append((k, v))
                else:
                    state_dict[k] = torch.tensor(v)

            for base, parts in chunks.items():
                parts.sort(key=lambda x: int(x[0].rsplit("__chunk_", 1)[1]))
                rows = []
                for _, v in parts:
                    rows.extend(v)
                state_dict[base] = torch.tensor(rows)

            self.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: torch.Tensor = None,
        token_type_ids: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Full forward pass: tokens -> pooled, normalised sentence embedding.

        Args:
            input_ids:      [batch, seq_len]   integer token IDs (0–30521)
            attention_mask: [batch, seq_len]   1 = real token, 0 = padding  (optional)
            token_type_ids: [batch, seq_len]   segment IDs, all-zeros for single sentences (optional)

        Returns:
            embedding: [batch, hidden_size]    L2-normalised sentence embedding
        """
        embedding_output = self.embeddings(input_ids, token_type_ids=token_type_ids)

        encoder_attention_mask = None
        if attention_mask is not None:
            encoder_attention_mask = (
                (1.0 - attention_mask.float()).unsqueeze(1).unsqueeze(2) * -1e9
            )

        sequence_output = self.encoder(embedding_output, encoder_attention_mask)
        if attention_mask is None:
            attention_mask = torch.ones(
                input_ids.size(0), input_ids.size(1),
                dtype=torch.float,
                device=input_ids.device,
            )

        return mean_pooling(sequence_output, attention_mask)

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int  = 64,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
        """
        Encode one or more sentences into fixed-size embeddings.

        Drop-in replacement for SentenceTransformer.encode() accepts the same
        arguments and returns the same output format.

        Args:
            sentences:            A single string or a list of strings.
            batch_size:           Number of sentences processed in one forward pass.
            show_progress_bar:    Show a tqdm progress bar over batches.
            convert_to_numpy:     Return numpy array (True) or torch.Tensor (False).
            normalize_embeddings: L2-normalise the output vectors.
                                  forward() always normalises, so this has no effect
                                  unless set to False (which skips the extra pass).

        Returns:
            If a single string was passed: array/tensor of shape [hidden_size].
            If a list was passed:          array/tensor of shape [N, hidden_size].
        """
        single = isinstance(sentences, str)
        if single:
            sentences = [sentences]

        all_embeddings: List[torch.Tensor] = []

        batches = range(0, len(sentences), batch_size)
        if show_progress_bar:
            try:
                from tqdm import tqdm
                batches = tqdm(batches, desc='Encoding', total=len(sentences) // batch_size)
            except ImportError:
                pass

        was_training = self.training
        self.eval()

        with torch.no_grad():
            for start in batches:
                batch_sentences = sentences[start : start + batch_size]
                input_ids, attention_mask = self.tokenizer.batch_tokenize(batch_sentences)
                all_embeddings.append(self.forward(input_ids, attention_mask))

        if was_training:
            self.train()

        result = torch.cat(all_embeddings, dim=0)
        if single:
            result = result.squeeze(0)

        return result.cpu().numpy() if convert_to_numpy else result
