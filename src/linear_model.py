import dotenv

dotenv.load_dotenv()

import logging

import torch



log = logging.getLogger(__name__)


class LinearModel(torch.nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        epochs: int, 
        learning_rate: float, 
        scheduler_factor: float, 
        scheduler_patience: int
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.linear = torch.nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x: torch.Tensor):
        return self.linear(x)
