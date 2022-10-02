import dotenv

dotenv.load_dotenv()

import hydra
import torch
from hydra import utils
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch import nn
from tqdm import (
    trange,
)

from src import losses
from src.linear_model import LinearModel
from src.utils import (
    create_vectorizers,
    load_data,
    set_seed,
)


@hydra.main(version_base=None, config_path="config", config_name="linear_model")
def main(config: DictConfig):
    config.seed = set_seed(config.seed)

    input_model, output_model = create_vectorizers(
        config.language, config.gpu, config.input_model.id, config.output_model.id
    )
    X, y = load_data(config.language, input_model, output_model)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    input_dim = X.shape[1]
    output_dim = y.shape[1]

    print(input_model.device)

    X_train = torch.tensor(X_train).to(input_model.device)
    X_test = torch.tensor(X_test).to(input_model.device)
    y_train = torch.tensor(y_train).to(input_model.device)
    y_test = torch.tensor(y_test).to(input_model.device)

    model: LinearModel = utils.instantiate(
        config.model, input_dim=input_dim, output_dim=output_dim
    )

    if torch.cuda.is_available():
        model.cuda()
    model = model.to(input_model.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=model.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        factor=model.scheduler_factor,
        patience=model.scheduler_patience,
        mode="min",
        verbose=True,
    )

    pbar = trange(model.epochs)
    for epoch in pbar:
        pbar.set_description(f"Epoch {epoch}")
        pbar.refresh()
        model.train()

        optimizer.zero_grad()
        output = model(X_train)
        training_loss = criterion(output, y_train)
        training_loss.backward()
        optimizer.step()

        model.eval()
        with torch.inference_mode():
            output = model(X_test)
            test_loss = criterion(output, y_test)
        scheduler.step(test_loss)

    with open(file="score.txt", mode="w", encoding="utf8") as f:
        score = str(losses.euclidean_distance(model(X_test), y_test).item())
        f.write(score)
    torch.save(model, "linear.pt")


if __name__ == "__main__":
    main()
