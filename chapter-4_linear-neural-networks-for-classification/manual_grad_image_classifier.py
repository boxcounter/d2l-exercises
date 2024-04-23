# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments

import math
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from loguru import logger


class FashionMNIST:
    labels = (
        't-shirt', 'trouser', 'pullover', 'dress', 'coat',
        'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
    )

    def __init__(
        self,
        batch_size: int = 64,
        resize: tuple[int, int] = (28, 28)
    ) -> None:
        self.batch_size = batch_size
        transform = transforms.Compose(
            [transforms.Resize(resize), transforms.ToTensor()])
        self.train: Dataset = datasets.FashionMNIST(
            root=".", train=True, transform=transform, download=True)
        self.valid: Dataset = datasets.FashionMNIST(
            root=".", train=False, transform=transform, download=True)

    def get_data_loader(self, train: bool = True) -> DataLoader:
        """
        Each row is a list, in which:
            #0 is tensor X, its shape = n * 1 * width * height
            #1 is tensor y, its shape = n * 1, values are indices like [9, 2, ...]
        """
        dataset = self.train if train else self.valid
        return DataLoader(dataset, self.batch_size, shuffle=train)

    def text_labels(self, indices: torch.Tensor) -> list[str]:
        return [self.labels[int(i)] for i in indices]

    def one_hot_labels(self, indices: torch.Tensor) -> torch.Tensor:
        rows = len(indices)
        labels = torch.zeros(size=(rows, len(self.labels)))
        for row in range(rows):
            column = indices[row]
            labels[row][column] = 1.0
        return labels

    def show_images(
        self,
        images: torch.Tensor,
        num_rows: int,
        num_cols: int,
        titles: list[str] | None = None,
        scale: float = 2.0
    ) -> None:
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, image) in enumerate(zip(axes, images)):
            image = image.numpy().squeeze()
            ax.imshow(image, cmap='gray')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles is not None:
                ax.set_title(titles[i])
        plt.show()

    def visualize(
        self,
        batch: list[torch.Tensor],
        ncols: int = 8,
    ) -> None:
        X: torch.Tensor | None = None
        labels = []

        if len(batch) == 3:
            X, y, y_pred = batch
            labels = [l + '\n' + pl
                      for l, pl in zip(self.text_labels(y),
                                       self.text_labels(y_pred))]
        elif len(batch) == 2:
            X, y = batch
            labels = self.text_labels(y)
        else:
            logger.error("Invalid batch length (expected 2 or 3, got {})", len(batch))
            return

        X = X.squeeze(1)  # Remove channel dimension for grayscale image
        nrows = math.ceil(len(X) / ncols)
        self.show_images(X, nrows, ncols, titles=labels)


class ManualGradClassifierModel(nn.Module):
    def __init__(
        self,
        learning_rate: float,
        num_channels: int,
        width: int,
        height: int,
        num_classes: int,
    ) -> None:
        super().__init__()

        self._lr = learning_rate
        self._weights = torch.normal(
            mean=0,
            std=random.random(),
            size=(width * height * num_channels,
                  num_classes) # shape = num_features * num_classes
            )
        self._bias = torch.zeros(
            size=(num_classes,)
        )
        self._logits: torch.Tensor | None = None

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Output shape = num_rows * num_classes
        """
        self._logits = X @ self._weights + self._bias
        return nn.functional.softmax(self._logits, dim=1)

    def backward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
    ) -> None:
        assert self._logits is not None, \
            "No logits to backpropagate. Call forward() first."
        assert self._logits.shape == y.shape, \
            f"Shapes mismatch between logits({self._logits.shape}) and y({y.shape})."

        #
        # Shape of cross_entropy_grad
        #   = num_rows * num_classes
        #
        cross_entropy_grad = self._logits - y
        batch_size = X.shape[0]

        #
        # Shape of weights_grad
        #   = X.T(num_features * num_rows) @ cross_entropy_grad(num_rows * num_classes)
        #   = num_features * num_classes
        # matches the shape of self._weights
        #
        weights_grad = X.T @ cross_entropy_grad / batch_size
        self._weights -= self._lr * weights_grad

        # The deriviate of logits with respect to bias is 1.
        bias_grad = cross_entropy_grad.sum(dim=0) / batch_size
        self._bias -= self._lr * bias_grad


class TransformMixin:
    def flatten(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(len(X), -1)

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return self._dataset.one_hot_labels(y) # type: ignore

    def index(self, y: torch.Tensor) -> torch.Tensor:
        return y.argmax(dim=1)


class Trainer(TransformMixin):
    def __init__(
        self,
        model: ManualGradClassifierModel,
        dataset: FashionMNIST,
        loss_measurer: nn.CrossEntropyLoss,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss_measurer = loss_measurer

    def train(self) -> float:
        num_batches = 0
        loss = 0.0
        for X, y in self._dataset.get_data_loader(train=True):
            X_ = self.flatten(X)
            y_ = self.one_hot(y)

            y_pred = self._model(X_)
            self._model.backward(X_, y_)

            loss += self._loss_measurer(y_pred, y_)
            num_batches += 1

        return loss / num_batches


class Samples:
    def __init__(
        self,
        max_count: int,
    ) -> None:
        self._DEFAULT_TENSOR = torch.Tensor([-1])
        self._X = self._DEFAULT_TENSOR
        self._y = self._DEFAULT_TENSOR
        self._y_pred = self._DEFAULT_TENSOR
        self._max_count = max_count

    def add(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> None:
        self._add_value("_X", X)
        self._add_value("_y", y.reshape(-1, 1))
        self._add_value("_y_pred", y_pred.reshape(-1, 1))

    def tensors(self) -> list[torch.Tensor]:
        return [
            self._X,
            self._y,
            self._y_pred,
        ]

    def _add_value(
        self,
        attr_name: str,
        value: torch.Tensor,
    ) -> None:
        new = None
        old = getattr(self, attr_name, self._DEFAULT_TENSOR)
        assert isinstance(old, torch.Tensor), \
            f"invalid attribute type (expected 'torch.Tensor', got {type(old)})"

        if len(old) >= self._max_count:
            return

        if old is self._DEFAULT_TENSOR:
            new = value
        else:
            new = torch.cat((old, value))

        setattr(self, attr_name, new)

    def __or__(self, other: "Samples") -> "Samples":
        result = Samples(self._max_count + other._max_count)
        result.add(self._X, self._y, self._y_pred)
        result.add(other._X, other._y, other._y_pred)
        return result


class Evaluator(TransformMixin):
    def __init__(
        self,
        model: ManualGradClassifierModel,
        dataset: FashionMNIST,
        loss_measurer: nn.CrossEntropyLoss,
        num_samples: int = 8,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss_measurer = loss_measurer
        self._loss = 0.0
        self._accuracy = 0.0
        self._corrects = Samples(num_samples)
        self._wrongs = Samples(num_samples)

    def evaluate(self) -> None:
        num_batches = 0
        loss = 0.0
        correct = 0
        total = 0

        for X, y_indices in self._dataset.get_data_loader(train=False):
            X_flatten = self.flatten(X)
            y_one_hot = self.one_hot(y_indices)

            y_pred = self._model(X_flatten)

            loss += self._loss_measurer(y_pred, y_one_hot)
            num_batches += 1

            y_pred_indices = self.index(y_pred)
            correct += (y_pred_indices == y_indices).sum().item()
            total += len(y_indices)

            self._collect_samples(X, y_indices, y_pred_indices)

        self._loss = loss / num_batches
        self._accuracy = correct / total

    @property
    def samples(self) -> list[torch.Tensor]:
        return (self._corrects | self._wrongs).tensors()

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def accuracy(self) -> float:
        return self._accuracy

    def _collect_samples(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> None:
        assert y_pred.shape == y.shape, \
            f"Shape mismatch, y_pred.shape = {y_pred.shape}, y.shape = {y.shape}"
        assert len(X) == len(y_pred), \
            f"Length mismatch, len(X) = {len(X)}, len(y_pred) = {len(y_pred)}"

        for x, y_, y_p_ in zip(X, y, y_pred):
            samples = self._corrects if torch.equal(y_, y_p_) else self._wrongs
            samples.add(x, y_, y_p_)


class LossPlotter:
    def __init__(self) -> None:
        self._epochs = []
        self._train_losses = []
        self._evaluate_losses = []

    def add(
        self,
        epoch: int,
        train_loss: float,
        evaluate_loss: float
    ) -> None:
        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._evaluate_losses.append(evaluate_loss)

    def plot(self) -> None:
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.plot(self._epochs, self._train_losses, 'b', label='train_loss')
        plt.plot(self._epochs, self._evaluate_losses, 'r', label='valid_loss')
        plt.legend()
        plt.title("FashionMNIST Classifier")
        plt.show()


def main(
    show_examples: bool = False
) -> None:
    num_channels = 1
    width = 32
    height = 32
    num_samples = 8
    max_epochs = 200
    learning_rate = 0.01

    dataset = FashionMNIST(resize=(width, height))
    if show_examples:
        batch = next(iter(dataset.get_data_loader(False)))
        dataset.visualize(batch[:16])

    num_classes = len(dataset.labels)
    model = ManualGradClassifierModel(
        learning_rate, num_channels, width, height, num_classes)

    loss_measurer = nn.CrossEntropyLoss()
    trainer = Trainer(model, dataset, loss_measurer)
    evaluator = Evaluator(model, dataset, loss_measurer, num_samples)
    plotter = LossPlotter()

    for epoch in range(max_epochs):
        train_loss = trainer.train()
        evaluator.evaluate()
        evaluate_loss = evaluator.loss
        accuracy = evaluator.accuracy
        logger.info("epoch #{}, train_loss = {}, evaluate_loss = {}, accuracy = {}",
                    epoch, train_loss, evaluate_loss, accuracy)
        plotter.add(epoch, train_loss, evaluate_loss)

    dataset.visualize(evaluator.samples)
    plotter.plot()

    logger.info("Done!")
    # Final output:
    # epoch #199, train_loss = 2.2297.., evaluate_loss = 2.2308..., accuracy = 0.7887


if __name__ == "__main__":
    main(True)
