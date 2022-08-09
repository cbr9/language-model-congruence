# -*- coding: utf-8 -*-

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import conllu
import requests
from tqdm import tqdm

from src import utils
from src.config import Config

log = logging.getLogger(f"{Path(__file__).name}:{__name__}")

dataset2url = {
    "french": "https://github.com/UniversalDependencies/UD_French-PUD/raw/master/fr_pud-ud-test.conllu",
    "spanish": "https://github.com/UniversalDependencies/UD_Spanish-PUD/raw/master/es_pud-ud-test.conllu",
    "swedish": "https://github.com/UniversalDependencies/UD_Swedish-PUD/raw/master/sv_pud-ud-test.conllu",
    "russian": "https://github.com/UniversalDependencies/UD_Russian-PUD/raw/master/ru_pud-ud-test.conllu",
    "portuguese": "https://github.com/UniversalDependencies/UD_Portuguese-PUD/raw/master/pt_pud-ud-test.conllu",
    "polish": "https://github.com/UniversalDependencies/UD_Polish-PUD/raw/master/pl_pud-ud-test.conllu",
    "korean": "https://github.com/UniversalDependencies/UD_Korean-PUD/raw/master/ko_pud-ud-test.conllu",
}


@dataclass
class Sentence:
    tokens: List[str]
    sentence_id: str


@dataclass
class Dataset:
    name: str
    train: List[conllu.TokenList]
    test: List[conllu.TokenList]

    def extract_sentences(self, token_type: str) -> Tuple[List[Sentence], List[Sentence]]:
        train_sentences = []
        test_sentences = []
        for sent in tqdm(
            self.train,
            desc=f"Extracting training sentences for dataset '{self.name}'",
        ):
            train_sentences.append(
                Sentence(
                    tokens=[token[token_type] for token in sent],
                    sentence_id=sent.metadata["sent_id"],
                )
            )
        for sent in tqdm(
            self.test, desc=f"Extracting test sentences for dataset '{self.name}'"
        ):
            test_sentences.append(
                Sentence(
                    tokens=[token[token_type] for token in sent],
                    sentence_id=sent.metadata["sent_id"],
                )
            )
        return train_sentences, test_sentences


class DataLoader:
    def __init__(self, config: Config) -> None:

        self.config = config

        self.dataset = utils.path("data") / f"{self.config.language}.conllu"
        self.dataset.parent.mkdir(parents=True, exist_ok=True)

        if not self.dataset.exists():
            self.download(dataset=self.config.language)

    def load(self, train_size: float = 0.8) -> Dataset:
        parsed = conllu.parse(self.dataset.read_text(encoding="utf8"))
        threshold = int(len(parsed) * train_size)
        return Dataset(
            name=self.config.language,
            train=parsed[:threshold],
            test=parsed[threshold:],
        )

    def download(self, dataset: str) -> None:
        r = requests.get(dataset2url[dataset], stream=True)
        filename = f"data/{dataset}.conllu"

        with open(file=filename, mode="wb") as f:
            pbar = tqdm(
                desc=f"Downloading dataset '{dataset}'",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                total=int(r.headers["Content-Length"]),
            )
            pbar.clear()  # clear 0% info
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()
