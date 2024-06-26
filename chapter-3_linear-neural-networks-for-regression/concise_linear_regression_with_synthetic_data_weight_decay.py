# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments

from typing import Iterator, Protocol
import math
import random

import torch
import matplotlib.pyplot as plt
from loguru import logger


class DataLoader(Protocol):
    def get_batch(
        self,
        train: bool
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        raise NotImplementedError


class SyntheticLinearDataLoader(DataLoader):
    def __init__(
        self,
        w: list[float],
        b: float,
        mean_noise: float = 0.01,
        num_train: int = 1000,
        num_valid: int = 1000,
        batch_size: int = 32
    ) -> None:
        self.batch_size = batch_size

        assert len(w) == 2, f"Expecting w to be a 1D torch.Tensor of size 2, got {w}"
        assert isinstance(b, float), f"Expecting b to be a float, got {b!r}"

        feature_dim = len(w)
        self.X_train = torch.randn(num_train, feature_dim)
        self.X_valid = torch.randn(num_valid, feature_dim)

        noise = mean_noise * torch.randn(num_train)
        w_t = torch.Tensor(w)
        self.y_train = (self.X_train @ w_t + b + noise).reshape(-1, 1)
        self.y_valid = (self.X_valid @ w_t + b).reshape(-1, 1)

    def get_batch(
        self,
        train: bool
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        X = self.X_train if train else self.X_valid
        y = self.y_train if train else self.y_valid

        batch_size = self.batch_size
        num_batches = math.ceil(len(X) / batch_size)
        for i in range(num_batches):
            start = i * batch_size
            stop = min(start + batch_size, len(X))
            yield X[start:stop], y[start:stop]


class WeightDecayedLinearRegressionModel(torch.nn.Module):
    def __init__(
        self,
        features_dim: int,
        learning_rate: float,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self._net = torch.nn.Linear(features_dim, 1)
        self._net.weight.data.normal_(mean = 0, std=random.random())
        self._net.bias.data.fill_(1)
        self._optimizer = torch.optim.SGD(
            params=[self._net.weight, self._net.bias],
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self._net.forward(X)

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()
        self._optimizer.step()
        self._optimizer.zero_grad()

    @property
    def weights(self) -> list[float]:
        return self._net.weight.data.detach().tolist()

    @property
    def bias(self) -> float:
        return self._net.bias.data.detach().item()

    @property
    def l2_norm(self) -> float:
        return torch.norm(self._net.weight.data).item()


class Trainer:
    def __init__(
        self,
        data_loader: DataLoader,
        model: WeightDecayedLinearRegressionModel,
    ) -> None:
        self._dataloader = data_loader
        self._model = model
        self._loss_measurer = torch.nn.MSELoss()
        self._losses: list[float] = []

    def train(
        self,
    ) -> None:
        num_batches = 0
        total_loss = torch.tensor(0.0)
        for X, y in self._dataloader.get_batch(True):
            self._model.train()
            y_pred = self._model(X)
            loss = self._loss_measurer(y_pred, y)
            self._model.backward(loss)

            total_loss += loss
            num_batches += 1

        mean_loss = total_loss.item() / num_batches
        self._losses.append(mean_loss)

    @property
    def losses(self) -> list[float]:
        return self._losses


class Evaluator:
    def __init__(
        self,
        data_loader: DataLoader,
        model: WeightDecayedLinearRegressionModel,
    ) -> None:
        self._dataloader = data_loader
        self._model = model
        self._loss_measurer = torch.nn.MSELoss()
        self._losses: list[float] = []

    def evaluate(self) -> float:
        self._model.eval()
        with torch.no_grad():
            num_batches = 0
            loss = torch.tensor(0.0)
            for X, y in self._dataloader.get_batch(False):
                y_pred = self._model(X)
                loss += self._loss_measurer(y_pred, y)
                num_batches += 1

            mean_loss = loss.item() / num_batches
            self._losses.append(mean_loss)
            return mean_loss

    @property
    def losses(self) -> list[float]:
        return self._losses


def plot_losses(
    trainer: Trainer,
    evaluator: Evaluator,
    weight_decay: float,
) -> None:
    assert len(trainer.losses) == len(evaluator.losses), \
        "Trainer and Evaluator losses length mismatch"
    epochs = list(range(1, len(trainer.losses)+1))

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(epochs, trainer.losses, 'b', label='train_loss')
    plt.plot(epochs, evaluator.losses, 'r', label='valid_loss')
    plt.legend()
    plt.title(f"weight decay = {weight_decay}")
    plt.show()


def main():
    w = [13.9, 12.1]
    b = -1.9
    epochs = 30
    weight_decay = 0.2

    data_loader = SyntheticLinearDataLoader(w, b)
    model = WeightDecayedLinearRegressionModel(
        features_dim=len(w), learning_rate=0.01, weight_decay=weight_decay)
    trainer = Trainer(data_loader, model)
    evaluator = Evaluator(data_loader, model)

    for e in range(epochs):
        trainer.train()
        loss = evaluator.evaluate()
        logger.debug("epoch #{}, loss = {}", e, loss)

    logger.info("Before training: w = {}, b = {}", w, b)
    logger.info("After  training: w = {}, b = {}", model.weights, model.bias)
    logger.info("L2 norm of weights: {}", model.l2_norm)

    plot_losses(trainer, evaluator, weight_decay)

    logger.info("Done!")


if __name__ == "__main__":
    main()
