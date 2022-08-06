# -*- coding: utf-8 -*-

from dataclasses import dataclass
from collections import defaultdict
import logging
from typing import Dict, List, Union
from pathlib import Path
import conllu

import requests
from tqdm import tqdm

from src.config import Config, TokenType


log = logging.getLogger(f"{Path(__file__).name}:{__name__}")

dataset2url = {
    "french": "https://github.com/UniversalDependencies/UD_French-PUD/raw/master/fr_pud-ud-test.conllu",
    "spanish": "https://github.com/UniversalDependencies/UD_Spanish-PUD/raw/master/es_pud-ud-test.conllu",
    "swedish": "https://github.com/UniversalDependencies/UD_Swedish-PUD/raw/master/sv_pud-ud-test.conllu",
    "russian": "https://github.com/UniversalDependencies/UD_Russian-PUD/raw/master/ru_pud-ud-test.conllu",
    "portuguese": "https://github.com/UniversalDependencies/UD_Portuguese-PUD/raw/master/pt_pud-ud-test.conllu",
    "polish": "https://github.com/UniversalDependencies/UD_Polish-PUD/raw/master/pl_pud-ud-test.conllu",
    "korean": "https://github.com/UniversalDependencies/UD_Korean-PUD/raw/master/ko_pud-ud-test.conllu"
}

@dataclass
class Dataset:
    name: str
    train: List[conllu.TokenList]
    test: List[conllu.TokenList]

    def extract_sentences(self, config: Config):
        train_sentences = [token[config.token_type.value] for sent in tqdm(self.train, desc=f"Extracting training sentences for dataset '{self.name}'") for token in sent]
        test_sentences = [token[config.token_type.value] for sent in tqdm(self.test, desc=f"Extracting test sentences for dataset '{self.name}'") for token in sent]
        return train_sentences, test_sentences

class DataLoader:
    def __init__(self, config: Config) -> None:

        self.config = config

        data = Path(__file__).parent.parent.joinpath("data")
        data.parent.mkdir(parents=True, exist_ok=True)
        for dataset in config.datasets:
            path = data.joinpath(dataset).with_suffix(".conllu")
            if not path.exists():
                self.download(dataset=dataset)

    def load(self) -> Dict[str, Dataset]:
        parses = defaultdict(dict)
        for dataset in self.config.datasets:
            with open(f"data/{dataset}.conllu", mode="r") as f:
                data = f.read()
                parsed = conllu.parse(data)
                cutoff = int(len(parsed)*0.8)
                parses[dataset] = Dataset(
                    name=dataset, 
                    train=parsed[:cutoff], 
                    test=parsed[cutoff:]
                )
        return dict(parses)

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