import dotenv

dotenv.load_dotenv()

import logging

import hydra
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import src.losses as losses
import src.utils as utils

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="pca")
def main(config: DictConfig):
    config.seed = utils.set_seed(config.seed)
    input_model, output_model = utils.create_vectorizers(
        config.language, config.gpu, config.input_model.id, config.output_model.id
    )
    X, y = utils.load_data(config.language, input_model, output_model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


    min_dim = min(X.shape[1], y.shape[1])
    pca = PCA(n_components=min_dim)

    if X.shape[1] != min_dim:
        pca.fit(X_train)
        X_test = pca.fit_transform(X_test)
    elif y.shape[1] != min_dim:
        pca.fit(y_train)
        y_test = pca.transform(y_test)

    with open(file="score.txt", mode="w", encoding="utf8") as f:
        score = str(losses.euclidean_distance(X_test, y_test))
        f.write(score)

if __name__ == "__main__":
    main()
