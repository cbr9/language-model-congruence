import dotenv

dotenv.load_dotenv()

import hydra
from omegaconf import DictConfig
from sklearn.manifold import TSNE
import seaborn as sns

import src.losses as losses
import src.utils as utils


@hydra.main(version_base=None, config_path="config", config_name="tsne")
def main(config: DictConfig):
    config.seed = utils.set_seed(config.seed)
    input_model, output_model = utils.create_vectorizers(
        config.language, config.gpu, config.input_model.id, config.output_model.id
    )
    X, y = utils.load_data(config.language, input_model, output_model)

    min_dim = min(X.shape[1], y.shape[1])
    tsne_x = TSNE(n_components=2, learning_rate="auto", init="pca")
    X = tsne_x.fit_transform(X)

    tsne_y = TSNE(n_components=2, learning_rate="auto", init="pca")
    y = tsne_x.fit_transform(y)

    with open(file="score.txt", mode="w", encoding="utf8") as f:
        score = str(losses.euclidean_distance(X, y))
        print(score)
        f.write(score)

if __name__ == "__main__":
    main()