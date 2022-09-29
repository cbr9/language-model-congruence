from pydantic.dataclasses import dataclass
from typing import Tuple

@dataclass
class Config:
    token_type: str
    language: str
    language_models: Tuple[str, str]