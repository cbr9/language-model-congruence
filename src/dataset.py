# -*- coding: utf-8 -*-

import logging
from pydantic import BaseModel, PrivateAttr
from pathlib import Path
from typing import List, Literal

import conllu
import requests
from tqdm import tqdm

from src import utils

log = logging.getLogger(f"{Path(__file__).name}:{__name__}")

dataset2url = {
    "french": "https://github.com/UniversalDependencies/UD_French-PUD/raw/master/fr_pud-ud-test.conllu",
    "arabic": "https://github.com/UniversalDependencies/UD_Arabic-PUD/raw/master/ar_pud-ud-test.conllu",
    "german": "https://github.com/UniversalDependencies/UD_German-PUD/raw/master/de_pud-ud-test.conllu",
    "spanish": "https://github.com/UniversalDependencies/UD_Spanish-PUD/raw/master/es_pud-ud-test.conllu",
    "swedish": "https://github.com/UniversalDependencies/UD_Swedish-PUD/raw/master/sv_pud-ud-test.conllu",
    "russian": "https://github.com/UniversalDependencies/UD_Russian-PUD/raw/master/ru_pud-ud-test.conllu",
    "portuguese": "https://github.com/UniversalDependencies/UD_Portuguese-PUD/raw/master/pt_pud-ud-test.conllu",
    "polish": "https://github.com/UniversalDependencies/UD_Polish-PUD/raw/master/pl_pud-ud-test.conllu",
    "korean": "https://github.com/UniversalDependencies/UD_Korean-PUD/raw/master/ko_pud-ud-test.conllu",
    "english": "https://github.com/UniversalDependencies/UD_English-PUD/raw/master/en_pud-ud-test.conllu",
    "hindi": "https://github.com/UniversalDependencies/UD_Hindi-PUD/raw/master/hi_pud-ud-test.conllu",
    "finnish": "https://github.com/UniversalDependencies/UD_Finnish-PUD/raw/master/fi_pud-ud-test.conllu",
    "turkish": "https://github.com/UniversalDependencies/UD_Turkish-PUD/raw/master/tr_pud-ud-test.conllu",
    "thai": "https://github.com/UniversalDependencies/UD_Thai-PUD/raw/master/th_pud-ud-test.conllu",
    "italian": "https://github.com/UniversalDependencies/UD_Italian-PUD/raw/master/it_pud-ud-test.conllu",
    "japanese": "https://github.com/UniversalDependencies/UD_Japanese-PUD/raw/master/ja_pud-ud-test.conllu",
    "indonesian": "https://github.com/UniversalDependencies/UD_Indonesian-PUD/raw/master/id_pud-ud-test.conllu",
    "icelandic": "https://github.com/UniversalDependencies/UD_Icelandic-PUD/raw/master/is_pud-ud-test.conllu",
    "czech": "https://github.com/UniversalDependencies/UD_Czech-PUD/raw/master/cs_pud-ud-test.conllu",
    "chinese": "https://github.com/UniversalDependencies/UD_Chinese-PUD/raw/master/zh_pud-ud-test.conllu",
}


class Sentence(BaseModel):
    tokens: List[str]
    sentence_id: str


class Dataset(BaseModel):
    language: Literal[
        "french",
        "spanish",
        "swedish",
        "russian",
        "portuguese",
        "portuguese",
        "polish",
        "korean",
        "english",
        "hindi",
        "finnish",
        "turkish",
        "thai",
        "japanese",
        "arabic",
        "german",
        "italian",
        "indonesian",
        "icelandic",
        "czech",
        "chinese",
    ]
    _path: Path = PrivateAttr(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        self._path = utils.path("data") / f"{self.language}.conllu"
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self.download()

    def get_sentences(self) -> List[Sentence]:
        sentences = []
        parsed = conllu.parse(self._path.read_text(encoding="utf8"))
        for sent in tqdm(
            parsed,
            desc=f"Extracting sentences for dataset '{self.language}'",
            leave=False,
        ):
            sentences.append(
                Sentence(
                    tokens=[token["lemma"] for token in sent],
                    sentence_id=sent.metadata["sent_id"],
                )
            )
        return sentences

    def download(self) -> None:
        r = requests.get(dataset2url[self.language], stream=True)
        filename = utils.path("data") / f"{self.language}.conllu"

        with open(file=filename, mode="wb") as f:
            f.write(r.content)
