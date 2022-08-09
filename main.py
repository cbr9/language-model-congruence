from omegaconf import DictConfig, OmegaConf
import hydra

from src.config import Config
from src.dataloader import DataLoader
from src.vectorizer import Vectorizer


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(config: DictConfig):
    config = Config(**OmegaConf.to_object(config))
    dataloader = DataLoader(config)
    vectorizers = [Vectorizer(id=config.models[i], language=config.language, gpu=config.gpu, token_type=config.token_type) for i in range(len(config.models))]

    dataset = dataloader.load()
    train_sentences, test_sentences = dataset.extract_sentences(config.token_type)
    sentences = train_sentences + test_sentences
    ids = [sent.sentence_id for sent in sentences]

    vectors = dict()

    
    for v in vectorizers:
        vectors[v.id] = v(sentences)
        print(vectors[v.id][ids[0]][0].shape)
        

if __name__ == "__main__":
    main()
