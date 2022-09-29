import dotenv

dotenv.load_dotenv()

import logging

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import src.losses as losses
import src.utils as utils
from src.config import Config
from src.dataset import DataLoader
from src.vectorizer import Vectorizer

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="pca")
def main(config: DictConfig):
    dataloader = DataLoader(config)
    dataset = dataloader.load()
    sentences = dataset.get_sentences(config.token_type)

    input_model = Vectorizer(
        id=config.language_models[0],
        language=config.language,
        gpu=config.gpu,
        token_type=config.token_type,
    )

    X = utils.normalize(input_model(sentences))

    del input_model

    output_model = Vectorizer(
        id=config.language_models[1],
        language=config.language,
        gpu=config.gpu,
        token_type=config.token_type,
    )

    y = utils.normalize(output_model(sentences))

    del output_model

    print(X.shape)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bigger, smaller = None, None
    # print(X_train.shape, y_train.shape)
    if X.shape[1] > y.shape[1]:
        bigger = X
        smaller = y
    elif X.shape[1] < y.shape[1]:
        bigger = y
        smaller = X

    if bigger is not None and smaller is not None:
        pca = PCA(n_components=smaller.shape[1])
        if np.array_equal(X, bigger):
            X = pca.fit_transform(X)
        else:
            y = pca.fit_transform(y)

    print(losses.euclidean_distance(X, y))
    # tsne = TSNE(2)
    # projected = [tsne.fit_transform(bigger[i].cpu().numpy()) for i in range(len(bigger))]
    # projected = [tsne.fit_transform(smaller[i].cpu().numpy()) for i in range(len(bigger))]
    # print(projected[0].shape)

    ### Train network
    # model = LinearMapper(
    #     input_dim=X_train.size(1), output_dim=y_train.size(1), config=config
    # )
    # if torch.cuda.is_available():
    #     model.cuda()

    # model = model.to(device)
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.linear_nn.learning_rate)

    # for epoch in range(config.linear_nn.epochs):
    #     model.train()

    #     optimizer.zero_grad()
    #     output = model(X_train)
    #     loss = criterion(output, y_train)
    #     loss.backward()
    #     optimizer.step()

    #     model.eval()
    #     with torch.inference_mode():
    #         output = model(X_test)
    #         test_loss = criterion(output, y_test)

    #     log.info(f"Epoch: {epoch}; training loss: {loss}; test loss: {test_loss}; euclidean_distance: {losses.euclidean_distance(output, y_test)}")

    # torch.save(model, "linear.pt")


if __name__ == "__main__":
    main()
