# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals

import dataclasses
import math
import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from loguru import logger
import torchinfo


device = (
    "cuda:1"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class FashionMNISTDataset:
    """FashionMNIST dataset with image resizing and visualization."""
    _labels = datasets.FashionMNIST.classes

    def __init__(
        self,
        batch_size: int = 64,
        resize: tuple[int, int] = (28, 28)
    ) -> None:
        self.batch_size = batch_size
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            ])
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

    @property
    def num_labels(self) -> int:
        return len(self._labels)

    def text_labels(self, indices: torch.Tensor) -> list[str]:
        """Returns the text labels for the given indices."""
        return [self._labels[int(i)] for i in indices]

    def one_hot_labels(self, indices: torch.Tensor) -> torch.Tensor:
        """Returns one-hot encoded labels for the given indices."""
        rows = len(indices)
        labels = torch.zeros(size=(rows, len(self._labels)))
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

        axes = axes.flatten()
        for i, (ax, image) in enumerate(zip(axes, images)):
            image = image.cpu().numpy().squeeze()
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
        filename: str | None = None,
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
        self.show_images(X, nrows, ncols, titles=labels, save_filename=filename)


@dataclasses.dataclass
class ConvParam:
    out_channels: int
    kernel_size: int
    stride: int = 1
    padding: int = 0


@dataclasses.dataclass
class PoolParam:
    kernel_size: int
    stride: int = 1
    padding: int = 0


class AlexNet(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int], # channels, height, width
        conv_pool_params: list[tuple[ConvParam, PoolParam | None]],
        linear_params: list[tuple[int, float]],
        num_labels: int,
    ) -> None:
        super().__init__()

        # Initialize the network
        self._net = nn.Sequential()
        for cp, pp in conv_pool_params:
            self._net.append(nn.LazyConv2d(**dataclasses.asdict(cp)))
            self._net.append(nn.ReLU())
            if pp is not None:
                self._net.append(nn.MaxPool2d(**dataclasses.asdict(pp)))

        self._net.append(nn.Flatten())

        for n, dp in linear_params:
            self._net.append(nn.LazyLinear(n))
            self._net.append(nn.ReLU())
            self._net.append(nn.Dropout(dp))

        self._net.append(nn.LazyLinear(num_labels))
        self._net.to(device)

        # Initialize weights
        tensors = torch.zeros((1, *input_shape)).to(device)
        self.forward(tensors)
        self._net.apply(self._weights_init_fn)

        torchinfo.summary(self._net, input_size=(1, *input_shape), device=device, verbose=2)

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return self._net(X)

    @staticmethod
    def _weights_init_fn(module: nn.Module):
        if isinstance(module, nn.Conv2d | nn.Linear):
            assert not isinstance(module.weight, nn.UninitializedParameter), \
                f"Uninitialized weight for module {module}"
            print(f"Initializing weights of {module}")
            nn.init.xavier_uniform_(module.weight)


class TransformMixin:
    def index(self, y: torch.Tensor) -> torch.Tensor:
        return y.argmax(dim=1)


class Trainer(TransformMixin):
    def __init__(
        self,
        model: AlexNet,
        dataset: FashionMNISTDataset,
        loss_measurer: nn.CrossEntropyLoss,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss_measurer = loss_measurer
        self._optimizer = optimizer

    def fit(self) -> float:
        num_batches = 0
        total_loss = 0.0

        self._model.train()
        for X, y in self._dataset.get_data_loader(train=True):
            X = X.to(device)
            y = y.to(device)

            y_logits_pred = self._model(X)
            loss = self._loss_measurer(y_logits_pred, y)

            # Backpropagation
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


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


class Validator(TransformMixin):
    def __init__(
        self,
        model: AlexNet,
        dataset: FashionMNISTDataset,
        loss_measurer: nn.CrossEntropyLoss,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss_measurer = loss_measurer

    def __call__(
        self,
        max_samples: int = 0,
    ) -> tuple[float, float, list[torch.Tensor]]:
        """Validates the model and returns loss, accuracy, and samples."""
        num_batches = 0
        total_loss = 0.0
        num_correct = 0
        num_total = 0
        corrects = Samples(int(max_samples/2))
        wrongs = Samples(int(max_samples/2))

        self._model.eval()
        with torch.inference_mode():
            for X, y_indices in self._dataset.get_data_loader(train=False):
                X = X.to(device)
                y_indices = y_indices.to(device)

                y_logits_pred = self._model(X)

                total_loss += self._loss_measurer(y_logits_pred, y_indices).item()
                num_batches += 1

                y_pred_indices = self.index(y_logits_pred)
                num_correct += (y_pred_indices == y_indices).sum().item()
                num_total += len(y_indices)

                self._collect_samples(corrects, wrongs, X, y_indices, y_pred_indices)

        mean_loss = total_loss / num_batches
        accuracy = num_correct / num_total
        samples = (corrects | wrongs).tensors()
        return (mean_loss, accuracy, samples)

    def _collect_samples(
        self,
        corrects: Samples,
        wrongs: Samples,
        X: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> None:
        assert y_pred.shape == y.shape, \
            f"Shape mismatch, y_pred.shape = {y_pred.shape}, y.shape = {y.shape}"
        assert len(X) == len(y_pred), \
            f"Length mismatch, len(X) = {len(X)}, len(y_pred) = {len(y_pred)}"

        for x, y_, y_p_ in zip(X, y, y_pred):
            samples = corrects if torch.equal(y_, y_p_) else wrongs
            samples.add(x, y_, y_p_)


class MetricsPlotter:
    def __init__(
        self,
        title: str,
        filename: str
    ) -> None:
        self._title = title
        self._filename = filename
        self._epochs = []
        self._train_losses = []
        self._evaluate_losses = []
        self._accuracies = []

    def add(
        self,
        epoch: int,
        train_loss: float,
        validate_loss: float,
        accuracy: float,
    ) -> None:
        assert isinstance(train_loss, float), \
            f"Invalid type for train_loss (expected 'float', got {type(train_loss)})"
        assert isinstance(validate_loss, float), \
            f"Invalid type for evaluate_loss (expected 'float', got {type(validate_loss)})"
        assert isinstance(accuracy, float), \
            f"Invalid type for accuracy (expected 'float', got {type(accuracy)}"

        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._evaluate_losses.append(validate_loss)
        self._accuracies.append(accuracy)

    def plot(self) -> None:
        fig, ax1 = plt.subplots()

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
        ax2.plot(self._epochs, self._accuracies, 'g', label='Validation Accuracy', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0, 1.0)
        ax2.legend(loc='upper right')

        plt.title(self._title)
        fig.savefig(self._filename)
        plt.show()


def main(
    preview_dataset: bool = True
) -> None:
    input_shape = (1, 224, 224)
    batch_size = 128
    num_samples = 16
    max_epochs = 20
    learning_rate = 0.01
    weight_decay = 0.01
    start_sec = time.time()

    # Initialize dataset
    dataset = FashionMNISTDataset(batch_size=batch_size, resize=input_shape[1:])
    if preview_dataset:
        batch = next(iter(dataset.get_data_loader(False)))
        dataset.visualize(batch)

    # Initialize model, loss, and optimizer
    model = AlexNet(
        input_shape=input_shape,
        conv_pool_params=[
            (
                ConvParam(out_channels=96, kernel_size=11, stride=4, padding=1),
                PoolParam(kernel_size=3, stride=2, padding=0),
            ),
            (
                ConvParam(out_channels=256, kernel_size=5, stride=1, padding=2),
                PoolParam(kernel_size=3, stride=2, padding=0),
            ),
            (
                ConvParam(out_channels=384, kernel_size=3, stride=1, padding=1),
                None,
            ),
            (
                ConvParam(out_channels=384, kernel_size=3, stride=1, padding=1),
                None,
            ),
            (
                ConvParam(out_channels=256, kernel_size=3, stride=1, padding=1),
                PoolParam(kernel_size=3, stride=2, padding=0)
            ),
        ],
        linear_params=[
            (4096, 0.5),
            (4096, 0.5),
        ],
        num_labels=10,
    )
    loss_measurer = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay)

    # Train and validate the model
    trainer = Trainer(model, dataset, loss_measurer, optimizer)
    validator = Validator(model, dataset, loss_measurer)
    plotter = MetricsPlotter(
        title="FashionMNIST Classifier (AlexNet)", filename="metrics.png")

    samples: list[torch.Tensor] = []
    for epoch in range(max_epochs):
        train_loss = trainer.fit()
        validate_loss, accuracy, samples = validator(
            num_samples if epoch == max_epochs - 1 else 0)

        logger.info("epoch #{}, train_loss = {:.3f}, validate_loss = {:.3f}, accuracy = {:.1%}",
                    epoch, train_loss, validate_loss, accuracy)
        plotter.add(epoch, train_loss, validate_loss, accuracy)

    logger.info("elapsed time: {:.1f} seconds", time.time() - start_sec)

    # Visualize samples (both correct and wrong predictions) from the last epoch
    dataset.visualize(samples, filename="pred_samples.png")
    plotter.plot()

    logger.info("done!")


if __name__ == "__main__":
    main(False)

    # Final output:
    # epoch #19, train_loss = 0.295, validate_loss = 0.308, accuracy = 88.6%
    # elapsed time: 560.0 seconds
    # done!
