from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

class TokenType(str, Enum):
    lemma = "lemma"
    form = "form"


@dataclass
class Config:
    gpu: Optional[int]
    
    token_type: str
    language: str
    models: Tuple[str, str]
