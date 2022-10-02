import dotenv

dotenv.load_dotenv()

import logging

import hydra
from omegaconf import DictConfig
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.linalg import orthogonal_procrustes
import src.losses as losses
import src.utils as utils

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="procrustes")
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
        X_train = pca.fit_transform(X_train)
    elif y.shape[1] != min_dim:
        y_train = pca.fit_transform(y_train)

    R, scale = orthogonal_procrustes(X_train, y_train)
    y_pred = X_test @ R

    with open(file="score.txt", mode="w", encoding="utf8") as f:
        score = str(losses.euclidean_distance(y_pred, y_test))
        f.write(score)

if __name__ == "__main__":
    main()
