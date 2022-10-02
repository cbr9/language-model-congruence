import uuid
from collections import defaultdict
from pathlib import Path
from typing import (
    List,
    Optional,
)

import numpy as np
import pandas as pd
import torch
import tqdm
from pandas import DataFrame
from pydantic import (
    BaseModel,
    PrivateAttr,
)
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    logging,
)

import src.utils as utils
from src.dataset import Sentence

logging.set_verbosity_error()


class Vectorizer(BaseModel):
    """
    id: huggingface identifier of a pretrained model (e.g., bert-base-uncased, xlm-roberta-base, etc.)
    """

    language: str
    token_type: str = "lemma"
    gpu: Optional[int]
    id: str

    _device: torch.device = PrivateAttr(default=None)
    _index: DataFrame = PrivateAttr(default=None)
    _model: PreTrainedModel = PrivateAttr(default=None)
    _tokenizer: PreTrainedTokenizerBase = PrivateAttr(default=None)
    _index_dir: Path = PrivateAttr(default=None)

    def __init__(
        self, **data
    ):
        super().__init__(**data)
        self._index_dir = utils.path(".cache")
        self._index_dir.mkdir(exist_ok=True)

    @property
    def device(
        self
    ):
        if self._device is None:
            self._device = torch.device(
                "cpu" if self.gpu is None else f"cuda:{self.gpu}"
            )
        return self._device

    @property
    def tokenizer(
        self
    ):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.id)
        return self._tokenizer

    @property
    def model(
        self
    ):
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.id).to(self.device)
            self._model.eval()
        return self._model

    @property
    def index(
        self
    ) -> DataFrame:
        if self._index is None:
            path = self._index_dir / "index.csv"
            if path.exists():
                self._index = pd.read_csv(path, engine="pyarrow")
            else:
                self._index = DataFrame(
                    columns=["model", "language", "id", "token_type"], )
            self.clean_cache()
        return self._index

    def index_row(
        self, identifier: str
    ):
        return DataFrame(
            [{
                "model": self.id, "language": self.language, "id": identifier, "token_type": self.token_type,
            }]
        )

    def clean_cache(
        self
    ):
        valid_ids = set(self._index.id.tolist())
        for file in self._index_dir.iterdir():
            if file.stem != "index" and file.stem not in valid_ids:
                file.unlink()

    def retrieve(self) -> np.ndarray | None:
        mask = ((self.index.model == self.id) & (self.index.language == self.language) & (
                self.index.token_type == self.token_type))
        row = self.index[mask]

        if not row.empty:
            assert len(row) == 1
            id_ = row.id.iloc[0]
            path = str(self._index_dir / f"{id_}.npy")
            try:
                return np.load(path)
            except FileNotFoundError:
                return None

        return None

    def store(
        self, embeddings: np.ndarray
    ) -> None:
        ids = self.index.id.tolist()
        while True:
            id_ = str(uuid.uuid4())
            if id_ not in ids:
                np.save(file=str(self._index_dir / f"{id_}.npy"), arr=embeddings)
                self._index = pd.concat(
                    [self.index, self.index_row(identifier=id_)], ignore_index=True, )
                path = self._index_dir / "index.csv"
                self._index.to_csv(path, index=False)
                break

    def __call__(
        self, sentences: List[Sentence]
    ) -> np.ndarray:

        word_embeddings = self.retrieve()
        if word_embeddings is None:
            word_embeddings = []
            batch_size = 100
            batches = [sentences[i: i + batch_size] for i in range(0, len(sentences), batch_size)]
            for batch in tqdm.tqdm(
                    batches, desc=f"Computing vectors for model {self.id}", leave=False
            ):
                tokens = [sent.tokens for sent in batch]
                batch_tokenization = self.tokenizer(
                    tokens, is_split_into_words=True, return_tensors="pt", truncation=True, padding=True, ).to(
                    self.device
                )

                with torch.no_grad():
                    batch_embeddings = self.model(
                        **batch_tokenization
                    ).last_hidden_state

                for i in range(len(batch)):
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

                    for word_id, embeddings in subword_embeddings_for_words.items():
                        word_embeddings.append(
                            torch.vstack(embeddings).mean(dim=0).cpu().numpy()
                        )
            word_embeddings = np.array(word_embeddings)
            self.store(word_embeddings)
        return word_embeddings
