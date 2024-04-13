# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


class FashionMNIST:
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
        dataset = self.train if train else self.valid
        return DataLoader(dataset, self.batch_size, shuffle=train)

    def text_labels(self, indices: torch.Tensor) -> list[str]:
        """Return text labels."""
        labels = [
            't-shirt', 'trouser', 'pullover', 'dress', 'coat',
            'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot'
        ]
        return [labels[int(i)] for i in indices]

    def show_images(
        self,
        imgs: torch.Tensor,
        num_rows: int,
        num_cols: int,
        titles: list[str] = None,
        scale: float = 1.5
    ) -> None:
        """Plot a list of images."""
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            img = img.numpy().squeeze()  # Convert tensor image to numpy
            ax.imshow(img, cmap='gray')
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles is not None:
                ax.set_title(titles[i])
        plt.show()  # Display the images

    def visualize(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        nrows: int = 1,
        ncols: int = 8,
        labels: list[str] = None
    ) -> None:
        X, y = batch
        X = X.squeeze(1)  # Remove channel dimension for grayscale image
        if labels is None:
            labels = self.text_labels(y)
        self.show_images(X, nrows, ncols, titles=labels)


def main() -> None:
    data = FashionMNIST(resize=(32, 32))
    batch = next(iter(data.get_data_loader(False)))
    data.visualize(batch)


if __name__ == "__main__":
    main()
