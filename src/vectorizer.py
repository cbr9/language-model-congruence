import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series
import tqdm
from transformers import AutoModel, AutoTokenizer, logging

from src.dataloader import Sentence

logging.set_verbosity_error()


@dataclass
class Vectorizer:
    """
    id: huggingface identifier of a pretrained model (e.g., bert-base-uncased, xlm-roberta-base, etc.)
    """

    gpu: Optional[int]
    id: str
    language: str
    token_type: str

    def __post_init__(self):
        self._device = None
        self._tokenizer = None
        self._model = None

    @property
    def device(self):
        if self._device is None:
            self._device = torch.device(
                "cpu" if self.gpu is None else f"cuda:{self.gpu}"
            )
        return self._device

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.id)
        return self._tokenizer

    @property
    def model(self):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.id).to(self.device)
            self._model.eval()
        return self._model

    def __call__(self, sentences: List[Sentence]) -> torch.Tensor:

        batch_size = 100
        batches = [
            sentences[i : i + batch_size]
            for i in range(0, len(sentences), batch_size)
        ]
        embeddings_for_sentences = defaultdict(list)
        for batch in tqdm.tqdm(batches, desc=f"Computing vectors for model {self.id}"):
            tokens = [sent.tokens for sent in batch]
            batch_tokenization = self.tokenizer(
                tokens,
                is_split_into_words=True,
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)

            with torch.no_grad():
                batch_embeddings = self.model(
                    **batch_tokenization
                ).last_hidden_state

            for i in tqdm.trange(len(batch), leave=False):
                sentence_embeddings = batch_embeddings[i]
                # word_ids look like [None, 0, 1, 1, 2, 3, 3, 4, None]
                word_ids = batch_tokenization.word_ids(batch_index=i)
                subword_embeddings_for_words = defaultdict(list)
                for j, word_id in enumerate(word_ids):
                    if word_id is None:
                        continue
                    subword_embeddings_for_words[word_id].append(
                        sentence_embeddings[j]
                    )
                embeddings_for_sentence = {}
                for word_id, embeddings in subword_embeddings_for_words.items():
                    embeddings_for_sentence[word_id] = (
                        torch.vstack(embeddings).mean(dim=0).cpu().numpy()
                    )
                embeddings_for_sentences[batch[i].sentence_id] = embeddings_for_sentence

        return embeddings_for_sentences
