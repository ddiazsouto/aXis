"""
We are using a BERT-like tokenizer used by NVIDIA's BERT implementation
and copied form their open source codebase. The original code can be found here:
https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/LanguageModeling/BERT/tokenization.py


The original tokenizer can be found here in python (hugginface has open source examples but they are all in rust):

https://github.com/google-research/bert/blob/master/tokenization.py
"""

import unicodedata
from pathlib import Path
from typing import List, Tuple

import torch

from .tokeniser import WordpieceTokenizer


class BertTokenizer:

    def __init__(self, model_path: Path, max_length: int = 128, unk_token: str = "[UNK]"):
        vocab_path = Path(model_path) / "vocab.txt"
        with open(vocab_path) as fh:
            self.vocab = {line.strip(): i for i, line in enumerate(fh)}

        self._unk_id    = self.vocab.get(unk_token, 0)
        self._max_length = max_length
        self._wp        = WordpieceTokenizer(vocab=self.vocab, unk_token=unk_token)

    def _basic_tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        buf = []
        for ch in text:
            if unicodedata.category(ch).startswith("P"):
                buf.append(f" {ch} ")
            else:
                buf.append(ch)
        return "".join(buf).split()

    def tokenize(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens = ["[CLS]"]
        for word in self._basic_tokenize(text):
            tokens.extend(self._wp.tokenize(word))
            if len(tokens) >= self._max_length - 1:
                break
        tokens.append("[SEP]")

        ids  = torch.tensor([[self.vocab.get(t, self._unk_id) for t in tokens]], dtype=torch.long)
        mask = torch.ones_like(ids)
        return ids, mask

    def batch_tokenize(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        tokenized = [self.tokenize(t) for t in texts]
        max_len   = max(ids.size(1) for ids, _ in tokenized)
        n         = len(tokenized)

        padded_ids  = torch.zeros(n, max_len, dtype=torch.long)
        padded_mask = torch.zeros(n, max_len, dtype=torch.long)

        for i, (ids, mask) in enumerate(tokenized):
            seq_len = ids.size(1)
            padded_ids[i,  :seq_len] = ids[0]
            padded_mask[i, :seq_len] = mask[0]

        return padded_ids, padded_mask
