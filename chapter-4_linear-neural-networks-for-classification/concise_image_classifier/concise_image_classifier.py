# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments
# pylint:disable=import-outside-toplevel

import math
from typing import TYPE_CHECKING

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from loguru import logger

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


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
        return DataLoader(dataset, self.batch_size, shuffle=train, pin_memory=True)#, pin_memory_device=device)

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


class ConciseClassifierModel(nn.Module):
    def __init__(
        self,
        num_classes: int,
        # num_channels: int,
        # width: int,
        # height: int,
    ) -> None:
        super().__init__()

        self._net = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(out_features=num_classes),
            # We don't need a softmax layer, as the loss function will take
            # care of it (cross_entropy uses "LogSumExp trick" internally).
        )

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Returns unnormalized logits instead of the softmax results.
        """
        return self._net(X)


class TransformMixin:
    def flatten(self, X: torch.Tensor) -> torch.Tensor:
        return X.reshape(len(X), -1)

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return self._dataset.one_hot_labels(y) # type: ignore

    def index(self, y: torch.Tensor) -> torch.Tensor:
        return y.argmax(dim=1)


def loss_fn(
    y_pred: torch.Tensor,
    y: torch.Tensor
) -> torch.Tensor:
    return nn.functional.cross_entropy(y_pred, y)


class Trainer(TransformMixin):
    def __init__(
        self,
        model: ConciseClassifierModel,
        dataset: FashionMNISTDataset,
        optimizer: torch.optim.Optimizer
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._optimizer = optimizer

    def fit(self) -> float:
        num_batches = 0
        total_loss = torch.zeros([]).to(device)

        self._model.train()
        for X, y in self._dataset.get_data_loader(train=True):
            X_ = X.to(device)
            y_indices = y.to(device)

            y_logits_pred = self._model(X_)
            loss = loss_fn(y_logits_pred, y_indices)

            # Backpropagation
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        return total_loss.item() / num_batches


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


class PredStdev:
    """Collects prediction (probability distribution) standard deviation."""
    def __init__(self) -> None:
        self._y_logits_pred: torch.Tensor | None = None

    def collect(
        self,
        y_logits_pred: torch.Tensor,
    ) -> None:
        if self._y_logits_pred is None:
            self._y_logits_pred = y_logits_pred.detach()
        else:
            self._y_logits_pred = torch.cat(
                (self._y_logits_pred, y_logits_pred))

    def get(self) -> tuple[float, float]:
        assert self._y_logits_pred is not None, "No predictions collected."
        y_pred = nn.functional.softmax(self._y_logits_pred, dim=1)
        sd = y_pred.std(dim=1)
        return (sd.mean().item(), sd.median().item())


class Evaluator(TransformMixin):
    def __init__(
        self,
        model: ConciseClassifierModel,
        dataset: FashionMNISTDataset,
        num_samples: int = 8,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss = 0.0
        self._accuracy = 0.0
        self._corrects = Samples(num_samples)
        self._wrongs = Samples(num_samples)

    def evaluate(self) -> None:
        num_batches = 0
        loss = 0.0
        correct = 0
        total = 0

        self._model.eval()
        with torch.no_grad():
            psd = PredStdev()
            for X, y in self._dataset.get_data_loader(train=False):
                X_ = X.to(device)
                y_indices = y.to(device)

                y_logits_pred = self._model(X_)

                loss += loss_fn(y_logits_pred, y_indices).item()
                num_batches += 1

                psd.collect(y_logits_pred)

                y_pred_indices = self.index(y_logits_pred)
                correct += (y_pred_indices == y_indices).sum().item()
                total += len(y)

                self._collect_samples(X, y_indices, y_pred_indices)

            self._loss = loss / num_batches
            self._accuracy = correct / total

            mean, median = psd.get()
            logger.debug("Prediction stdev: mean = {}, median = {}", mean, median)

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
        ax1.set_ylim(0)
        ax1.legend(loc='upper left')

        # Create a second y-axis for accuracy
        ax2 = ax1.twinx()
        ax2.set_ylabel('accuracy', color='tab:blue')
        ax2.plot(self._epochs, self._accuracies, 'g', label='Accuracy', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0, 1.0)
        ax2.legend(loc='upper right')

        plt.title("Concise FashionMNIST Classifier")
        fig.savefig("metrics.png")
        plt.show()


def main(
    preview_dataset: bool = True
) -> None:
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
    model = ConciseClassifierModel(num_classes).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    trainer = Trainer(model, dataset, optimizer)
    evaluator = Evaluator(model, dataset, num_samples)
    plotter = MetricsPlotter()

    for epoch in range(max_epochs):
        train_loss = trainer.fit()
        evaluator.evaluate()
        logger.info("epoch #{}, train_loss = {}, evaluate_loss = {}, accuracy = {}",
                    epoch, train_loss, evaluator.loss, evaluator.accuracy)
        plotter.add(epoch, train_loss, evaluator.loss, evaluator.accuracy)

    dataset.visualize(evaluator.samples, save_filename="pred_samples.png")
    plotter.plot()

    logger.info("Done!")
    # Final output:
    # epoch #49, train_loss = 0.4404, evaluate_loss = 0.4723, accuracy = 0.833


if __name__ == "__main__":
    main(False)
