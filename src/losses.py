from functools import singledispatch

import numpy as np
import torch


@singledispatch
def euclidean_distance(
    y_pred, y
):
    raise NotImplementedError


@euclidean_distance.register
def euclidean_distance_tensors(
    y_pred: torch.Tensor, y: torch.Tensor
):
    n_rows = y_pred.size(0)
    distances = torch.tensor([torch.norm(y[r] - y_pred[r]) for r in range(n_rows)])
    return torch.mean(distances)


@euclidean_distance.register
def euclidean_distance_arrays(
    y_pred: np.ndarray, y: np.ndarray
):
    distances = np.linalg.norm(y - y_pred, axis=1)
    return np.mean(distances)
