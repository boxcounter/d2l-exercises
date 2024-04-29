# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments
# pylint:disable=import-outside-toplevel

import math
import random
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from loguru import logger


class FashionMNISTDataset:
    """FashionMNIST dataset with image resizing and visualization."""
    labels = datasets.FashionMNIST.classes

    def __init__(
        self,
        batch_size: int = 64,
        resize: tuple[int, int] = (28, 28)
    ) -> None:
        self.batch_size = batch_size
        transform = transforms.Compose(
            [transforms.Resize(resize), transforms.ToTensor()])
        self.train: Dataset = datasets.FashionMNIST(
            root="..", train=True, transform=transform, download=True)
        self.valid: Dataset = datasets.FashionMNIST(
            root="..", train=False, transform=transform, download=True)

    def get_data_loader(self, train: bool = True) -> DataLoader:
        """
        Returns a DataLoader object for the dataset.

        Each DataLoader object yields a tuple of two tensors:
        - X: Image tensor of shape (batch_size, 1, width, height)
        - y: Label tensor of shape (batch_size,)
        """
        dataset = self.train if train else self.valid
        return DataLoader(dataset, self.batch_size, shuffle=train)

    def text_labels(self, indices: torch.Tensor) -> list[str]:
        """Returns the text labels for the given indices."""
        return [self.labels[int(i)] for i in indices]

    def one_hot_labels(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns one-hot encoded labels for the given indices."""
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
        scale: float = 2.0,
        row_spacing: float = 0.5,
        save_filename: str | None = None,
    ) -> None:
        figsize = (num_cols * scale, num_rows * scale)
        fig, axes = plt.subplots(num_rows,
                                 num_cols,
                                 figsize=figsize,
                                 gridspec_kw={'hspace': row_spacing})

        if TYPE_CHECKING:
            # Using type-checking to suppress pyright error when using fig, like:
            # error: Cannot access attribute "savefig" for class "FigureBase"
            assert isinstance(fig, plt.Figure)

        axes = axes.flatten()
        for i, (ax, image) in enumerate(zip(axes, images)):
            image = image.numpy().squeeze()
            ax.imshow(image, cmap='gray')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles is not None:
                ax.set_title(titles[i])

        plt.show()
        if save_filename is not None:
            fig.savefig(save_filename)

    def visualize(
        self,
        batch: list[torch.Tensor],
        ncols: int = 8,
        save_filename: str | None = None,
    ) -> None:
        X: torch.Tensor | None = None
        labels = []

        if len(batch) == 3:
            X, y, y_pred = batch
            labels = ['Real: ' + l + '\n' + 'Pred: ' + pl
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
        self.show_images(X, nrows, ncols, titles=labels, save_filename=save_filename)


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
        Forward pass of the model.

        Returns unnormalized logits instead of the softmax results.
        """
        return X @ self._weights + self._bias

    def backward(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_logits_pred: torch.Tensor,
    ) -> None:
        assert y_logits_pred.shape == y.shape, \
            f"Shapes mismatch between logits({y_logits_pred.shape}) and y({y.shape})."

        #
        # Shape of cross_entropy_grad
        #   = num_rows * num_classes
        #
        cross_entropy_grad = y_logits_pred - y
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
        dataset: FashionMNISTDataset,
        loss_measurer: nn.CrossEntropyLoss,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss_measurer = loss_measurer

    def train(self) -> float:
        num_batches = 0
        loss = 0.0
        for X, y in self._dataset.get_data_loader(train=True):
            X_flatten = self.flatten(X)
            y_one_hot = self.one_hot(y)

            y_logits_pred = self._model(X_flatten)
            self._model.backward(X_flatten, y_one_hot, y_logits_pred)

            loss += self._loss_measurer(y_logits_pred, y_one_hot)
            num_batches += 1

        return loss / num_batches


class Samples:
    """Collects samples for correct and wrong predictions."""
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
            f"Invalid attribute type (expected 'torch.Tensor', got {type(old)})"

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
        dataset: FashionMNISTDataset,
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

            y_logits_pred = self._model(X_flatten)

            loss += self._loss_measurer(y_logits_pred, y_one_hot)
            num_batches += 1

            y_pred_indices = self.index(y_logits_pred)
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


class MetricsPlotter:
    def __init__(self) -> None:
        self._epochs = []
        self._train_losses = []
        self._evaluate_losses = []
        self._accuracies = []

    def add(
        self,
        epoch: int,
        train_loss: float,
        evaluate_loss: float,
        accuracy: float,
    ) -> None:
        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._evaluate_losses.append(evaluate_loss)
        self._accuracies.append(accuracy)

    def plot(self) -> None:
        fig, ax1 = plt.subplots()

        if TYPE_CHECKING:
            # Using type-checking to suppress pyright errors like:
            # error: Cannot access attribute "set_xlabel" for class "ndarray[Any, dtype[Any]]"
            from matplotlib.axes._axes import Axes
            assert isinstance(fig, plt.Figure)
            assert isinstance(ax1, Axes)

        # Plot losses on the first y-axis
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.plot(self._epochs, self._train_losses, 'b', label='Train Loss')
        ax1.plot(self._epochs, self._evaluate_losses, 'r', label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper left')

        # Create a second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy', color='tab:blue')
        ax2.plot(self._epochs, self._accuracies, 'g', label='Accuracy', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0, 1.0)
        ax2.legend(loc='upper right')

        plt.title("Manual-grad FashionMNIST Classifier")
        fig.savefig("metrics.png")
        plt.show()


def main(
    preview_dataset: bool = True
) -> None:
    num_channels = 1
    width = 24
    height = 24
    num_samples = 8
    max_epochs = 50
    learning_rate = 0.01

    dataset = FashionMNISTDataset(resize=(width, height))
    if preview_dataset:
        batch = next(iter(dataset.get_data_loader(False)))
        dataset.visualize(batch)

    num_classes = len(dataset.labels)
    model = ManualGradClassifierModel(
        learning_rate, num_channels, width, height, num_classes)

    loss_measurer = nn.CrossEntropyLoss()
    trainer = Trainer(model, dataset, loss_measurer)
    evaluator = Evaluator(model, dataset, loss_measurer, num_samples)
    plotter = MetricsPlotter()

    for epoch in range(max_epochs):
        train_loss = trainer.train()
        evaluator.evaluate()
        logger.info("epoch #{}, train_loss = {}, evaluate_loss = {}, accuracy = {}",
                    epoch, train_loss, evaluator.loss, evaluator.accuracy)
        plotter.add(epoch, train_loss, evaluator.loss, evaluator.accuracy)

    dataset.visualize(evaluator.samples, save_filename="pred_samples.png")
    plotter.plot()

    logger.info("Done!")
    # Final output:
    # epoch #49, train_loss = 1.791, evaluate_loss = 1.795, accuracy = 0.809


if __name__ == "__main__":
    main(False)
