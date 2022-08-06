from omegaconf import DictConfig, OmegaConf
import hydra

from src.config import Config, TokenType
from src.dataloader import DataLoader, dataset2url
from src.vectorizer import Vectorizer


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(config: DictConfig):
    config = Config(**OmegaConf.to_object(config))
    vectorizers = Vectorizer()
    dataloader = DataLoader(config)
    datasets = dataloader.load()


if __name__ == "__main__":
    main()