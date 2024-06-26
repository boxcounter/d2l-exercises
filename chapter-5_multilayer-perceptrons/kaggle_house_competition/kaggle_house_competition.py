# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals

import os
import sys
from zipfile import ZipFile
from typing import TYPE_CHECKING

from loguru import logger
from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import torch
from torch import nn, optim
import torch.utils
import torch.utils.data
import matplotlib.pylab as plt


ID_LABEL = 'Id'
Y_LABEL = 'SalePrice'


class KaggleHouseDateset:
    """Kaggle House Prices Competition Dataset"""
    def __init__(
        self,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        num_folds: int = 5,
    ) -> None:
        train_df, test_df = self._preprocess_data(train_data, test_data)
        self._train_folds = self._fold_data(train_df, num_folds)
        self._test_data = self._tensor_from_pandas(test_df)

    def get_data_loaders(
        self,
        valid_fold_index: int,
        batch_size: int,
    ) -> tuple[torch.utils.data.DataLoader,
               torch.utils.data.DataLoader]:
        """
        Get data loaders for training and validation data.

        Returns:
            A tuple of training and validation data loaders.
        """
        logger.debug("creating data loaders, index of valid fold = {}", valid_fold_index)
        assert valid_fold_index < len(self._train_folds)

        train_features: list[torch.Tensor] = []
        train_targets: list[torch.Tensor] = []
        valid_features: torch.Tensor = torch.zeros((0))
        valid_target: torch.Tensor = torch.zeros((0))
        for i, sample in enumerate(self._train_folds):
            if i == valid_fold_index:
                valid_features = sample[0]
                valid_target = sample[1]
            else:
                train_features.append(sample[0])
                train_targets.append(sample[1])

        # train data loader
        dataset = torch.utils.data.TensorDataset(
            torch.cat(train_features), torch.cat(train_targets))
        train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        # validate data loader
        dataset = torch.utils.data.TensorDataset(valid_features, valid_target)
        valid_data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

        logger.debug("batches: train = {}, valid = {}",
                     len(train_data_loader), len(valid_data_loader))
        return (train_data_loader, valid_data_loader)

    def get_test_data(self) -> torch.Tensor:
        return self._test_data

    @staticmethod
    def _preprocess_data(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Preprocess the data.

        Returns:
            A tuple of preprocessed training and test data.
        """
        assert train_data is not None
        assert test_data is not None

        # Combine train and test data to preprocess them together, otherwise the
        # one-hot encoding may not be consistent.
        train_features = train_data.drop(columns=[ID_LABEL, Y_LABEL])
        test_features = test_data.drop(columns=ID_LABEL)
        features = pd.concat((train_features, test_features))

        # Normalize features
        numeric_column_labels = features.select_dtypes(include='number').columns
        numeric_columns = features[numeric_column_labels]
        features[numeric_column_labels] = numeric_columns.apply(
            lambda x: (x - x.mean()) / x.std())

        # NAN to 0
        features = features.fillna(0)

        # Categorical to one-hot
        features = pd.get_dummies(features, dtype=int)

        start_of_test = train_features.shape[0]
        train_df = features[:start_of_test].copy().assign(**{Y_LABEL: train_data[Y_LABEL]})
        test_df = features[start_of_test:].copy()
        return (train_df, test_df)

    @staticmethod
    def _fold_data(
        data: pd.DataFrame,
        num_folds: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Split the data into multiple folds.

        Returns:
            A list of folds, each fold contains a tuple of features and target tensors.
        """
        assert num_folds > 1
        logger.debug(f"splitting data({data.shape}) into {num_folds} folds")

        folds = []
        fold_size = data.shape[0] // num_folds

        for i in range(num_folds):
            start_row = i * fold_size
            stop_row = None if (i == num_folds - 1) else start_row + fold_size
            rows = data.iloc[start_row:stop_row]

            logger.debug(f"fold {i}: {rows.shape}")

            tensors = KaggleHouseDateset._tensors_from_pandas(rows)
            folds.append(tensors)

            logger.debug(f"fold {i}: {tensors[0].shape}, {tensors[1].shape}")

        return folds

    @staticmethod
    def _tensors_from_pandas(
        data: pd.DataFrame
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert pandas DataFrame to tensors.

        Returns:
            A tuple of features and target tensors.
        """
        assert Y_LABEL in data

        features = data.drop(columns=Y_LABEL)
        target = data[Y_LABEL]
        return (
            KaggleHouseDateset._tensor_from_pandas(features),
            # Use the logarithm helps normalize the differences and allows comparison
            # of relative errors more effectively.
            torch.log(KaggleHouseDateset._tensor_from_pandas(target).reshape(-1, 1)),
        )

    @staticmethod
    def _tensor_from_pandas(
        data: pd.DataFrame | pd.Series
    ) -> torch.Tensor:
        """Convert pandas DataFrame or Series to a tensor."""
        return torch.tensor(data.values, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(
        self,
        num_hidden_units: list[int],
        dropout_probabilites: list[float],
    ) -> None:
        assert len(num_hidden_units) == len(dropout_probabilites)

        logger.debug("creating a model with {} hidden layers: hidden units = {}, dropout = {}",
                     len(num_hidden_units), num_hidden_units, dropout_probabilites)

        super().__init__()

        self._net = nn.Sequential()
        for n, p in zip(num_hidden_units, dropout_probabilites):
            self._net.append(nn.LazyLinear(n))
            self._net.append(nn.ReLU())
            self._net.append(nn.Dropout(p))
        self._net.append(nn.LazyLinear(1))

        logger.debug("model: {}", self._net)

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        return self._net(X)


class Trainer:
    def __init__(
        self,
        model: MLP,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        self._model = model
        self._loss_measurer = nn.MSELoss()
        self._optimizer = optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def __call__(
        self,
        data_loader: torch.utils.data.DataLoader,
    ) -> float:
        num_batches = 0
        total_loss = 0.0

        self._model.train(True)
        for X, y in data_loader:
            y_pred = self._model(X)

            loss = self._loss_measurer(y_pred, y)
            loss.backward()
            self._optimizer.step()
            self._optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


class Validator:
    def __init__(
        self,
        model: MLP,
    ) -> None:
        self._model = model
        self._loss_measurer = nn.MSELoss()

    def __call__(
        self,
        data_loader: torch.utils.data.DataLoader,
    ) -> float:
        num_batches = 0
        total_loss = 0.0

        self._model.train(False)
        with torch.inference_mode():
            for X, y in data_loader:
                y_pred = self._model(X)

                total_loss += self._loss_measurer(y_pred, y).item()
                num_batches +=1

        return total_loss / num_batches


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
        self._validate_losses = []

    def add(
        self,
        epoch: int,
        train_loss: float,
        validate_loss: float,
    ) -> None:
        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._validate_losses.append(validate_loss)

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
        ax1.plot(self._epochs, self._train_losses, 'b', label='Average train Loss')
        ax1.plot(self._epochs, self._validate_losses, 'r', label='Average validation Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper right')

        plt.title(self._title)
        fig.savefig(self._filename)
        plt.show()


def prepare_dataset(
    competition: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Download and prepare the dataset.

    Returns:
        A tuple of training and test data DataFrames.
    """
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition)

    filename = competition + ".zip"
    dirname = competition
    with ZipFile(filename) as z:
        z.extractall(dirname)

    train_file = competition + "/train.csv"
    assert os.path.isfile(train_file), f"'train.csv' not found in '{filename}'"

    test_file = competition + "/test.csv"
    assert os.path.isfile(test_file), f"'test.csv' not found in '{filename}'"

    return (pd.read_csv(train_file), pd.read_csv(test_file))


def k_fold(
    dataset: KaggleHouseDateset,
    k: int,
    max_epochs: int,
    plotter: MetricsPlotter,
) -> list[MLP]:
    """
    Perform k-fold training and cross-validation.

    Returns:
        A list of trained models.
    """
    learning_rate = 0.01
    weight_decay = 0.01
    num_hidden_units = [64] * 6
    dropout_probabilities = [0.12] * 6

    models = []
    trainers = []
    validators = []
    for _ in range(k):
        model = MLP(num_hidden_units, dropout_probabilities)
        models.append(model)
        trainers.append(Trainer(model, learning_rate, weight_decay))
        validators.append(Validator(model))

    batch_size = 64
    for epoch in range(max_epochs):
        total_train_loss = 0.0
        total_validate_loss = 0.0
        for i, trainer, validator in zip(range(k), trainers, validators):
            train_data_loader, valid_data_loader = dataset.get_data_loaders(i, batch_size)

            train_loss = trainer(train_data_loader)
            validate_loss = validator(valid_data_loader)

            logger.debug("epoch #{} - fold #{}: train_loss = {:.3f}, validate_loss = {:.3f}",
                        epoch, i, train_loss, validate_loss)

            total_train_loss += train_loss
            total_validate_loss += validate_loss

        average_train_loss = total_train_loss / k
        average_validate_loss = total_validate_loss / k

        plotter.add(epoch, average_train_loss, average_validate_loss)

        logger.info("epoch #{}: average_train_loss = {:.3f}, average_validate_loss = {:.3f}",
                    epoch, average_train_loss, average_validate_loss)

    return models


def predict(
    models: list[MLP],
    features: torch.Tensor
) -> torch.Tensor:
    """
    Predict the target values.

    Returns:
        A tensor of predicted target values with the shape (n, 1).
    """
    prediction = [model(features) for model in models]
    prediction = torch.exp(
        torch.cat(prediction, dim=1).mean(dim=1, keepdim=True)
    ).detach()

    logger.debug("prediction: max = {}, mean = {}, median = {}, std = {}",
                 prediction.max(), prediction.mean(), prediction.median(), prediction.std())
    return prediction


def save_prediction(
    ids: pd.Series,
    pred: torch.Tensor,
    filename: str,
) -> None:
    pd.DataFrame({
        ID_LABEL: ids,
        Y_LABEL: pred.squeeze().numpy(),
        }).to_csv(filename, index=False)


def main():
    competition = "house-prices-advanced-regression-techniques"
    train_df, test_df = prepare_dataset(competition)
    logger.debug(train_df[Y_LABEL].describe())

    dataset = KaggleHouseDateset(train_df, test_df)
    plotter = MetricsPlotter("Kaggle House Competition", "metrics.png")

    # K-fold training and cross-validation
    num_folds = 5
    max_epochs = 30
    models = k_fold(dataset, num_folds, max_epochs, plotter)
    plotter.plot()

    # Predict
    test_tensors = dataset.get_test_data()
    prices_pred = predict(models, test_tensors)
    save_prediction(test_df[ID_LABEL], prices_pred, "submission.csv")

    logger.info("Done!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sink=sys.stderr, level="INFO")

    main()
