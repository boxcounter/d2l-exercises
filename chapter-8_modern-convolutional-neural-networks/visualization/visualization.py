# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals

import sys
import math
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms, datasets, utils
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)
import matplotlib.pyplot as plt
from loguru import logger
import torchinfo


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


class CIFAR10Dataset:
    """CIFAR10 dataset with image resizing and visualization."""
    _labels = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    ]

    def __init__(
        self,
        batch_size: int = 64,
        resize: tuple[int, int] = (28, 28)
    ) -> None:
        self.batch_size = batch_size
        self.mean = (0.5, 0.5, 0.5)
        self.std = (0.5, 0.5, 0.5)
        transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
            ])

        self.train: Dataset = datasets.CIFAR10(
            root="..", train=True, transform=transform, download=True)
        self.valid: Dataset = datasets.CIFAR10(
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

    def denormalize(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        tensors = images.clone()
        for t, m, s in zip(tensors, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensors

    def show_images(
        self,
        titled_images: list[tuple[str, torch.Tensor]],
        num_cols: int,
        scale: float = 1.2,
        row_spacing: float = 1.0,
        denormailize: bool = True,
        filename: str | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        titles = [ti[0] for ti in titled_images]
        images = [ti[1] for ti in titled_images]

        num_rows = math.ceil(len(images) / num_cols)
        figsize = CIFAR10Dataset._calculate_figure_size(
            titled_images, num_rows, num_cols, scale)
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=figsize, gridspec_kw={'hspace': row_spacing})

        axes = axes.flatten()
        i = 0
        for i, (ax, title, image) in enumerate(zip(axes, titles, images)):
            if denormailize:
                image = self.denormalize(image)
            image = image.cpu().numpy().transpose((1, 2, 0))  # Transpose to (H, W, C)
            image = np.clip(image, 0, 1)  # Clip values to [0, 1]

            if image.shape[-1] == 1: # grayscale image in (H, W, 1)
                ax.imshow(image.squeeze(-1), cmap='gray')
            elif image.shape[-1] == 3: # RGB
                ax.imshow(image)
            else:
                raise ValueError(f"Invalid image shape: {image.shape}")

            # Hide axes
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_title(title)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.show()

        if filename is not None:
            fig.savefig(filename)

        if writer is not None:
            writer.add_figure(filename if filename is not None else ' ', fig)

    def visualize(
        self,
        batch: list[torch.Tensor],
        ncols: int = 8,
        filename: str | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        if len(batch) == 3:
            titled_images = []
            for x, y, y_pred in zip(*batch):
                if y == y_pred:
                    title = f"✔︎ {self.text_labels(y)[0]}"
                else:
                    title = f"✖︎ {self.text_labels(y_pred)[0]}"

                titled_images.append((title, x))
        elif len(batch) == 2:
            X, y = batch
            titled_images = list(zip(self.text_labels(y), X))
        else:
            raise ValueError(f"Invalid batch length (expected 2 or 3), got {len(batch)}")

        self.show_images(titled_images, ncols, filename=filename, writer=writer)

    @staticmethod
    def _calculate_figure_size(
        titled_images: list[tuple[str, torch.Tensor]],
        num_rows: int,
        num_cols: int,
        scale: float,
    ) -> tuple[float, float]:
        max_height = 0.0
        max_width = 0.0
        for _, image in titled_images:
            max_height = max(max_height, image.shape[1])
            max_width = max(max_width, image.shape[2])
        return (num_cols * scale * max_width / 100,
                num_rows * scale * max_height / 100)


class SampleCollection:
    """Collects samples for correct and wrong predictions."""
    def __init__(
        self,
        max_samples: int = 8,
    ) -> None:
        self._corrects = SampleCollection.Samples(name='corrects', max_count=max_samples//2)
        self._wrongs = SampleCollection.Samples(name='wrongs', max_count=max_samples//2)

    def add(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        y_pred: torch.Tensor,
    ) -> None:
        assert y_pred.shape == y.shape, \
            f"Shape mismatch, y_pred.shape = {y_pred.shape}, y.shape = {y.shape}"
        assert len(X) == len(y_pred), \
            f"Length mismatch, len(X) = {len(X)}, len(y_pred) = {len(y_pred)}"

        correct_indices = y == y_pred
        self._corrects.add(
            X[correct_indices], y[correct_indices], y_pred[correct_indices])

        wrong_indices = y != y_pred
        self._wrongs.add(
            X[wrong_indices], y[wrong_indices], y_pred[wrong_indices])

    def tensors(self) -> list[torch.Tensor]:
        return (self._corrects | self._wrongs).tensors()

    class Samples:
        def __init__(
            self,
            name: str,
            max_count: int,
        ) -> None:
            self._EMPTY_TENSOR = torch.empty(0)
            self._name = name
            self._X = self._EMPTY_TENSOR
            self._y = self._EMPTY_TENSOR
            self._y_pred = self._EMPTY_TENSOR
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
            return [self._X, self._y, self._y_pred]

        def _add_value(
            self,
            attr_name: str,
            value: torch.Tensor,
        ) -> None:
            old = getattr(self, attr_name, self._EMPTY_TENSOR)
            remaining_count = self._max_count - len(old)
            if remaining_count <= 0:
                return

            if old is self._EMPTY_TENSOR:
                new = value[:remaining_count]
            else:
                new = torch.cat((old, value[:remaining_count]))

            setattr(self, attr_name, new)

        def __or__(
            self,
            other: "SampleCollection.Samples"
        ) -> "SampleCollection.Samples":
            result = SampleCollection.Samples(
                name='all', max_count=self._max_count + other._max_count)
            result.add(self._X, self._y, self._y_pred)
            result.add(other._X, other._y, other._y_pred)
            return result


class ResNeXtBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        num_groups: int,
        intermediate_channels: int,
        stride: int = 1,
        use_1x1conv: bool = False,
    ) -> None:
        super().__init__()

        self._mapping = nn.Sequential(
            nn.LazyConv2d(out_channels=intermediate_channels,
                          kernel_size=1,
                          stride=1,
                          groups=num_groups),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=out_channels,
                          kernel_size=3,
                          stride=stride,
                          padding=1,
                          groups=num_groups),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=intermediate_channels,
                          kernel_size=1,
                          stride=1,
                          groups=num_groups),
            nn.LazyBatchNorm2d(),
        )

        self._connection = nn.Sequential(
            nn.LazyConv2d(out_channels, kernel_size=1, stride=stride)
                if use_1x1conv
                else ResNeXtBlock._PassthroughConnection(),
            nn.LazyBatchNorm2d(),
        )

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        return nn.functional.relu(self._mapping(X) + self._connection(X))

    class _PassthroughConnection(nn.Module):
        def forward(
            self,
            X: torch.Tensor
        ) -> torch.Tensor:
            return X


class ResNeXt(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int], # channels, height, width
        out_channels: list[int],
        intermediate_channels: list[int],
        num_groups: int,
        num_labels: int,
    ) -> None:
        super().__init__()

        # Initialize the network
        self._net = nn.Sequential(
            self._new_stem(),
            self._new_body(out_channels, intermediate_channels, num_groups),
            self._new_head(num_labels),
        )

        self._net.to(device)

        # Initialize weights
        tensors = torch.zeros((1, *input_shape)).to(device)
        self.forward(tensors)
        self._net.apply(self._weights_init_fn)

        self._summary(input_shape)

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
            nn.init.xavier_uniform_(module.weight)

    def _new_stem(self) -> nn.Module:
        return nn.Sequential(
            nn.LazyConv2d(out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LazyBatchNorm2d(),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

    def _new_body(
        self,
        out_channels: list[int],
        intermediate_channels: list[int],
        num_groups: int,
    ) -> nn.Module:
        assert len(out_channels) > 0
        assert len(out_channels) == len(intermediate_channels)

        # Modules made up of residual blocks.
        net = nn.Sequential()

        # First module
        net.append(nn.Sequential(
            ResNeXtBlock(out_channels[0], num_groups, intermediate_channels[0]),
            ResNeXtBlock(out_channels[0], num_groups, intermediate_channels[0]),
            ))

        # Susequent modules
        for oc, ic in zip(out_channels[1:], intermediate_channels[1:]):
            net.append(nn.Sequential(
                ResNeXtBlock(out_channels=oc,
                             num_groups=num_groups,
                             intermediate_channels=ic,
                             stride=2,
                             use_1x1conv=True),
                ResNeXtBlock(out_channels=oc,
                             num_groups=num_groups,
                             intermediate_channels=ic),
            ))

        return net

    def _new_head(
        self,
        num_labels: int,
    ) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.LazyLinear(num_labels),
        )

    def _summary(
        self,
        input_shape: tuple[int, int, int], # channels, height, width
    ) -> None:
        columns = ("output_size", "kernel_size", "num_params", "params_percent")
        torchinfo.summary(
            self._net, input_size=(1, *input_shape), device=device, col_names=columns)


class ResNeXt18(ResNeXt):
    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_labels: int
    ) -> None:
        out_channels=[64, 128, 256, 512]
        intermediate_channels = [64, 128, 256, 512]
        num_groups = 8
        super().__init__(input_shape, out_channels, intermediate_channels, num_groups, num_labels)


class Trainer:
    def __init__(
        self,
        model: ResNeXt18,
        dataset: CIFAR10Dataset,
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


class Validator:
    def __init__(
        self,
        model: ResNeXt18,
        dataset: CIFAR10Dataset,
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
        samples = SampleCollection(max_samples)

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

                samples.add(X, y_indices, y_pred_indices)

        mean_loss = total_loss / num_batches
        accuracy = num_correct / num_total
        return (mean_loss, accuracy, samples.tensors())

    def index(self, y: torch.Tensor) -> torch.Tensor:
        return y.argmax(dim=1)


class MetricsPlotter:
    def __init__(
        self,
    ) -> None:
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

    def plot(
        self,
        title: str,
        filename: str,
        writer: SummaryWriter | None = None,
    ) -> None:
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

        plt.title(title)
        plt.show()

        if filename is not None:
            fig.savefig(filename)

        if writer is not None:
            writer.add_figure(filename if filename is not None else ' ', fig)


def even_sample(
    data: list | torch.Tensor,
    num_samples: int,
) -> tuple[list, int]:
    """Returns a list of evenly sampled elements from the data.

    Parameters:
    - data: The data to sample from.
    - num_samples: The number of samples to return.

    Returns:
    - A tuple containing the list of samples and the interval between samples.
    """
    if num_samples <= 0:
        return [], 0

    total_elements = len(data)

    if num_samples >= total_elements:
        return data if isinstance(data, list) else [*data], 1

    interval = total_elements // num_samples
    return [data[i*interval] for i in range(num_samples)], interval


class FeatureExtractor:
    def __init__(
        self,
        model: nn.Module,
        num_layers: int = 8,
    ) -> None:
        _, eval_nodes = get_graph_node_names(model)
        # Exclude the final two nodes (Flatten and Linear) that are not convolutional
        return_nodes, _ = even_sample(eval_nodes[:-2], num_layers)
        self._model = create_feature_extractor(model, return_nodes)

    def save(
        self,
        dataset: CIFAR10Dataset,
        image_index: int = 0,
        num_images_per_output: int = 4,
        filename: str | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        """
        Saves the output of each layer for the given image.
        The output is saved as a single image with multiple subplots.

        Parameters:
        - dataset: The dataset object.
        - image_index: The index of the image in the dataset to use.
        - num_images_per_output: The number of images to show per layer output.
        """

        X, _ = next(iter(dataset.get_data_loader(False)))
        image = X[image_index].unsqueeze(0).to(device) # shape of image: (1, C, H, W)

        with torch.no_grad():
            # outputs is a dictionary of {node: tensor with shape (1, C, H, W)}
            outputs = self._model(image)

        titled_images = [('input', image[0])]
        for node, output in outputs.items(): # shape of output: (1, C, H, W)
            images, _ = even_sample(output.transpose(0, 1), num_images_per_output)

            images = utils.make_grid(
                images, nrow=num_images_per_output, normalize=True, pad_value=1, scale_each=True)
            titled_images.append((node, images))

        dataset.show_images(titled_images, num_cols=1, filename=filename, writer=writer)


class FilterExtractor:
    _filters: list[torch.Tensor] = []

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        self._model = model

    def save(
        self,
        dataset: CIFAR10Dataset,
        num_filters: int = 8,
        num_images_per_filter: int = 32,
        filename: str | None = None,
        writer: SummaryWriter | None = None,
    ) -> None:
        """
        Saves the filters of the model as images.

        Parameters:
        - dataset: The dataset object used to save the images.
        - num_filters: The number of filters to save.
        - num_images_per_filter: The number of images to show per filter.
        """

        self._model.apply(FilterExtractor._collect_filters_fn)

        titled_images = []
        filters, step = even_sample(FilterExtractor._filters, num_filters)
        for i, f in enumerate(filters):
            title = f"Filter {i*step} (Kernel {f.shape[-2]}x{f.shape[-1]})"
            # Transforming the filter with the shape (Co, Ci, H, W) to (Co*Ci, 1, H, W),
            # treating each filter as a batch of images with a single channel each.
            f = f.reshape(-1, 1, f.shape[-2], f.shape[-1])
            images, _ = even_sample(f, num_images_per_filter)
            image = utils.make_grid(
                images, nrow=num_images_per_filter, normalize=True, pad_value=1, scale_each=True)
            titled_images.append((title, image))

        dataset.show_images(
            titled_images, num_cols=1, denormailize=False, filename=filename, writer=writer)

    @staticmethod
    def _collect_filters_fn(module: nn.Module):
        if isinstance(module, nn.Conv2d):
            FilterExtractor._filters.append(module.weight.clone().detach())


def main(
    preview_dataset: bool = True
) -> None:
    input_shape = (3, 96, 96)
    batch_size = 128
    num_samples = 16
    max_epochs = 15
    learning_rate = 0.1
    weight_decay = 1e-5
    start_sec = time.time()
    writer = SummaryWriter()

    # Initialize dataset
    dataset = CIFAR10Dataset(batch_size=batch_size, resize=input_shape[1:])
    if preview_dataset:
        batch = next(iter(dataset.get_data_loader(False)))
        batch = [batch[0][:num_samples], batch[1][:num_samples]]
        dataset.visualize(batch, filename='preview.png', writer=writer)

    # Initialize model, loss, and optimizer
    model = ResNeXt18(input_shape=input_shape, num_labels=10)
    writer.add_graph(model, torch.zeros((1, *input_shape)).to(device))
    loss_measurer = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate, weight_decay)

    # Train and validate the model
    trainer = Trainer(model, dataset, loss_measurer, optimizer)
    validator = Validator(model, dataset, loss_measurer)
    plotter = MetricsPlotter()

    samples: list[torch.Tensor] = []
    for epoch in range(max_epochs):
        # We only want samples from the final epoch
        max_samples = num_samples if epoch == max_epochs - 1 else 0

        train_loss = trainer.fit()
        validate_loss, accuracy, samples = validator(max_samples)

        logger.info("epoch #{}, train_loss = {:.3f}, validate_loss = {:.3f}, accuracy = {:.1%}",
                    epoch, train_loss, validate_loss, accuracy)
        plotter.add(epoch, train_loss, validate_loss, accuracy)
        writer.add_scalars(
            "loss", {"train": train_loss, "validate": validate_loss}, epoch)
        writer.add_scalar("accuracy", accuracy, epoch)

    logger.info("device = {}, elapsed time: {:.1f} seconds", device, time.time() - start_sec)

    # Visualize samples (both correct and wrong predictions) from the last epoch
    dataset.visualize(samples, filename="pred_samples.png", writer=writer)
    plotter.plot(
        title=f"CIFAR10 Classifier ({ResNeXt18.__name__})", filename="metrics.png", writer=writer)

    # Visualize the output of each layer
    FeatureExtractor(model).save(dataset, filename="features.png", writer=writer)

    # Visualize filters
    FilterExtractor(model).save(dataset, filename="filters.png", writer=writer)

    writer.close()
    logger.info("done!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sink=sys.stderr, level="DEBUG")

    main(True)

    # Final output:
    # epoch #14, train_loss = 0.037, validate_loss = 1.237, accuracy = 71.5%
    # device = mps, elapsed time: 699.7 seconds
    # done!
