from pathlib import Path
from functools import singledispatch
import hydra
import torch
import torch.nn.functional as F
import numpy as np
import numpy.typing as npt

from src.vectorizer import Vectorizer
from src.dataset import Dataset


def set_seed(seed: int | None) -> int:
    if seed is None:
        seed = np.random.randint(low=1, high=1000, size=1).item()
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def create_vectorizers(language: str, gpu: str | None, input_model: str, output_model: str) -> tuple[Vectorizer, Vectorizer]:
    return (
        Vectorizer(
            language=language, gpu=gpu, id=input_model
        ),
        Vectorizer(
            language=language, gpu=gpu, id=output_model
        )
    )

def load_data(language: str, input_model: Vectorizer, output_model: Vectorizer) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    dataset = Dataset(language=language)
    sentences = dataset.get_sentences()

    return (
        input_model(sentences),
        output_model(sentences)
    )


def path(path: str) -> Path:
    return Path(hydra.utils.to_absolute_path(path))

    
@singledispatch
def normalize(data):
    raise NotImplementedError
    
@normalize.register
def normalize_tensor(t: torch.Tensor) -> torch.Tensor:
    return F.normalize(t, p=2, dim=1)

@normalize.register
def normalize_array(t: np.ndarray) -> np.ndarray:
    return t / np.linalg.norm(t, axis=1, ord=2, keepdims=True)
