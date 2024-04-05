# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments

from typing import Iterator, Protocol, Callable
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


class LinearRegressionModel(torch.nn.Module):
    def __init__(
        self,
        features_dim: int,
        learning_rate: float,
    ) -> None:
        super().__init__()
        self.w: torch.Tensor = torch.normal(
            0, random.random(), (features_dim, 1), requires_grad=True)
        self.b: torch.Tensor = torch.zeros(1, requires_grad=True)
        self.lr = learning_rate

    def forward(self, X: torch.Tensor):
        return X @ self.w + self.b

    def backward(self, loss: torch.Tensor):
        # logger.debug("backwarding with loss={}, lr={} ...", loss, self.lr)

        loss.backward()

        assert self.w.grad is not None, "w doesn't has grad"
        assert self.b.grad is not None, "b doesn't has grad"

        with torch.no_grad():
            self.w -= self.lr * self.w.grad
            self.w.grad.zero_() # type: ignore

            self.b -= self.lr * self.b.grad
            self.b.grad.zero_() # type: ignore

        # logger.debug("backwarded: ∆w is zero = {}, ∆b is zero = {} ...",
                    #  torch.any(self.lr * self.w.grad),
                    #  torch.any(self.lr * self.b.grad))


def squared_error(
    y_pred: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    return ((y_pred - y)**2 / 2).mean()


class Trainer:
    def __init__(
        self,
        data_loader: DataLoader,
        model: LinearRegressionModel,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self._dataloader = data_loader
        self._model = model
        self._loss_func = loss_func

    def train(
        self,
    ) -> None:
        for X, y in self._dataloader.get_batch(True):
            self._model.train()
            y_pred = self._model(X)
            loss = self._loss_func(y_pred, y)
            self._model.backward(loss)


class Evaluator:
    def __init__(
        self,
        data_loader: DataLoader,
        model: LinearRegressionModel,
        loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> None:
        self._dataloader = data_loader
        self._model = model
        self._loss_func = loss_func
        self._losses: list[float] = []

    def evaluate(self) -> float:
        self._model.eval()
        with torch.no_grad():
            num_batches = 0
            loss = torch.tensor(0.0)
            for X, y in self._dataloader.get_batch(False):
                y_pred = self._model(X)
                loss += self._loss_func(y_pred, y)
                num_batches += 1

            mean_loss = loss.item() / num_batches
            self._losses.append(mean_loss)
            return mean_loss

    def plot(
        self,
    ) -> None:
        epoch = list(range(1, len(self._losses)+1))
        losses = self._losses

        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(epoch, losses)
        plt.show()


def main():
    w = [13.9, 12.1]
    b = -1.9
    epochs = 20

    data_loader = SyntheticLinearDataLoader(w, b)
    model = LinearRegressionModel(features_dim=len(w), learning_rate=0.01)
    trainer = Trainer(data_loader, model, squared_error)
    evaluator = Evaluator(data_loader, model, squared_error)

    for e in range(epochs):
        trainer.train()
        loss = evaluator.evaluate()
        logger.debug("epoch #{}, loss = {}", e, loss)

    logger.info("Before training: w = {}, b = {}", w, b)
    w, b = model.w.detach().reshape(1, -1).tolist()[0], model.b.item()
    logger.info("After  training: w = {}, b = {}", w, b)

    evaluator.plot()
    logger.info("Done!")


if __name__ == "__main__":
    main()
