# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments, too-many-instance-attributes
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals

import sys
import time
import re
from collections import Counter

import torch
from torch import nn, optim
import torch.utils
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.nn import functional as Func
import matplotlib.pyplot as plt
from loguru import logger
import requests


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def assert_dimensions(
    tensor_name: str,
    tensor,
    expected_dimensions: int,
) -> None:
    assert tensor is not None, f"{tensor_name} is None"
    assert isinstance(tensor, torch.Tensor), \
        f"Invalid type for {tensor_name} (expected 'torch.Tensor', got {type(tensor)})"
    assert len(tensor.shape) == expected_dimensions, \
        (f"Invalid dimensions for {tensor_name} (expected {expected_dimensions}, "
         f"got {len(tensor.shape)})")


def assert_shape(
    tensor_name: str,
    tensor,
    expected_shape: tuple[int, ...],
) -> None:
    assert tensor is not None, f"{tensor_name} is None"
    assert isinstance(tensor, torch.Tensor), \
        f"Invalid type for {tensor_name} (expected 'torch.Tensor', got {type(tensor)})"
    assert tensor.shape == expected_shape, \
        f"Invalid shape for {tensor_name} (expected {expected_shape}, got {tensor.shape})"


class Vocabulary:
    RESERVED_TOKENS = ('<unk>', )

    def __init__(
        self,
        tokens: list[str],
        min_freq: int = 0,
        reserved_tokens: tuple[str, ...] = (),
    ) -> None:
        assert isinstance(tokens, list)

        tokens_freq = Counter(tokens)
        for token in list(tokens_freq.keys()):
            if tokens_freq[token] < min_freq:
                del tokens_freq[token]

        for token in list((*self.RESERVED_TOKENS, *reserved_tokens)):
            tokens_freq[token] = 0

        self._tokens_freq = tokens_freq
        self._tokens = list(self._tokens_freq.keys())
        self._indices = {token: i for i, token in enumerate(self._tokens)}

    def __len__(self) -> int:
        return len(self._tokens)

    def __getitem__(
        self,
        word: str,
    ) -> int:
        return self._indices[word]

    def tokenize(
        self,
        words: list[str] | str,
    ) -> list[int]:
        assert isinstance(words, list | str)

        if isinstance(words, str):
            words = list(words)

        return [self[token] for token in words]

    def decode(
        self,
        indices: list[float],
    ) -> list[str]:
        return [self._tokens[int(t)] for t in indices]

    def decode_one_hot(
        self,
        one_hots: torch.Tensor,
    ) -> str:
        assert_shape("one_hots", one_hots, (len(self),))
        return self._tokens[one_hots.argmax(dim=-1)]

    def most_common(
        self,
        n: int | None = None
    ) -> list[tuple[str, int]]:
        return self._tokens_freq.most_common(n)

    @property
    def size(self) -> int:
        return len(self)

    def one_hot_encode(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        return Func.one_hot(X, len(self)).to(torch.float32)


class TimeMachineDataset:
    URL = "https://www.gutenberg.org/ebooks/35.txt.utf-8"

    def __init__(
        self,
        filepath: str,
        batch_size: int,
        num_steps: int,
        training_set_ratio: float = 0.8,
    ) -> None:
        try:
            with open(filepath, 'r', encoding='UTF-8') as f:
                content = f.read()

            if not content or len(content) == 0:
                raise Exception("Empty file")
        except Exception:
            logger.debug("downloading content from {}", self.URL)
            content = self._download(self.URL, filepath)

        logger.debug("content length = {}", len(content))

        self._batch_size = batch_size
        content = self._text_to_chars(content)
        self._vocab = Vocabulary(content)
        self._training_set, self._validation_set = self._construct_dataset(
            self._vocab, content, num_steps, training_set_ratio)

    def get_data_loader(
        self,
        train: bool = True,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Returns a DataLoader object for the dataset.
        """
        dataset = self._training_set if train else self._validation_set
        return DataLoader(dataset, self._batch_size, shuffle)

    @property
    def vocab(self) -> Vocabulary:
        return self._vocab

    @staticmethod
    def _construct_dataset(
        vocab: 'Vocabulary',
        content: list[str],
        num_steps: int,
        training_set_ratio: float,
    ) -> tuple[Dataset, Dataset]:
        """
        Construct datasets from the content.

        Parameters:
        - vocab: the vocabulary object
        - content: the content of the dataset
        - num_steps: the number of time steps
        - training_set_ratio: the ratio of the training set

        Returns a tuple of two datasets:
        - item #0: the training dataset
        - item #1: the validation dataset
        """

        assert 0.0 < training_set_ratio < 1.0

        tokens = vocab.tokenize(content)
        X, Y = TimeMachineDataset._preprocess_tokens(tokens, num_steps)
        assert X.shape[0] == Y.shape[0]

        total = X.shape[0]
        train_size = int(total * training_set_ratio)

        return TensorDataset(X[:train_size], Y[:train_size]), \
            TensorDataset(X[train_size:], Y[train_size:])

    @staticmethod
    def _preprocess_tokens(
        tokens: list[int],
        num_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Preprocess the tokens into a dataset.

        Parameters:
        - tokens: the list of tokens
        - num_steps: the number of steps

        Returns a tuple of two torch.Tensor:
        - item #0: the input tensor X with the shape of (num_steps, len(tokens) - num_steps)
        - item #1: the target tensor Y with the shape of (num_steps, len(tokens) - num_steps)

        An example of 3 steps:
                  +----+----+----+----+----+----+----+----+----+----+
          tokens: |    |    |    |    |    |    |    |    |    |    |
                  +----+----+----+----+----+----+----+----+----+----+
          step 1:  x00  x10  x20  x30  x40  x50  x60
          step 1:       y00  y10  y20  y30  y40  y50  y60

          step 2:       x01  x11  x21  x31  x41  x51  x61
          step 2:            y01  y11  y21  y31  y41  y51  y61

          step 3:            x02  x12  x22  x32  x42  x52  x62
          step 3:                 y02  y12  y22  y32  y42  y52  y62

          x10 is the input token at the first step of the second sequence.
          y10 is the prediction at the first step of the second sequence.

          x11 is the input token at the second step of the second sequence.
          y11 is the prediction at the second step of the second sequence.

          x12 is the token at the third step of the second sequence.
          y12 is the prediction at the third step of the second sequence.
        """

        X, Y = [], []
        for i in range(num_steps):
            x_start = i
            x_stop = x_start + len(tokens) - num_steps
            # The dtype is inferred from the data type of the tokens, which
            # is expected to be integers.
            X.append(torch.tensor(tokens[x_start:x_stop]))

            y_start = x_start + 1
            y_stop = x_stop + 1
            Y.append(torch.tensor(tokens[y_start:y_stop]))

        X = torch.stack(X).transpose(0, 1)
        Y = torch.stack(Y).transpose(0, 1)

        logger.debug("X.shape = {}, Y.shape = {}", X.shape, Y.shape)
        return X, Y

    @staticmethod
    def _text_to_chars(
        text: str
    ) -> list[str]:
        """
        Convert the text into a list of characters.
        """

        return list(re.sub('[^A-Za-z]+', ' ', text).lower())

    @staticmethod
    def _download(
        url: str,
        filepath: str,
    ) -> str:
        """
        Download the content from the URL and save it to the file.
        """

        resp = requests.get(url, timeout=3)
        resp.raise_for_status()

        with open(filepath, 'w', encoding='UTF-8') as f:
            f.write(resp.text)

        return resp.text


class LSTMLMConcise(nn.Module):
    def __init__(
        self,
        vocab: Vocabulary,
        num_hidden_units: int,
        batch_first: bool = True,
        grad_clip: float = 1.0,
    ) -> None:
        super().__init__()

        self._num_layers = 1
        self._num_hidden_units = num_hidden_units
        self._vocab = vocab
        self._grad_clip = grad_clip

        self._mem_cell = nn.LSTM(
            vocab.size, num_hidden_units, self._num_layers, batch_first=batch_first, device=device)
        self._output_layer = nn.LazyLinear(vocab.size, device=device)

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - inputs: the input tensor with the shape of (batch_size, num_steps)

        Returns the outputs with the shape of (batch_size, num_steps, vocab_size)
        """

        outputs, _ = self._forward(inputs, None)
        assert_shape('output', outputs, (inputs.shape[0], inputs.shape[1], self._vocab.size))
        return outputs

    def predict(
        self,
        prefix: str,
        num_prediction: int,
    ) -> str:
        """
        Predict the next tokens based on the prefix.

        Parameters:
        - prefix: the prefix string
        - num_prediction: the number of tokens to predict

        Returns the predicted string.
        """

        H_C = None
        num_layers, vocab_size, num_hidden_units = (
            self._num_layers, self._vocab.size, self._num_hidden_units)
        batch_size, num_steps = 1, 1

        # Warm up
        for char in prefix[:-1]:
            tokens = self._vocab.tokenize(char)
            inputs = torch.tensor(tokens, device=device).unsqueeze(0) # batch_size = 1
            assert_shape('inputs', inputs, (batch_size, num_steps))
            _, H_C = self._forward(inputs, H_C)

        # Predict
        sequence = [prefix[-1]]
        for _ in range(num_prediction):
            tokens = self._vocab.tokenize(sequence[-1])
            inputs = torch.tensor(tokens, device=device).unsqueeze(0)
            assert_shape('inputs', inputs, (batch_size, num_steps))

            outputs, H_C = self._forward(inputs, H_C)
            assert_shape('outputs', outputs, (batch_size, num_steps, vocab_size))
            assert_shape('H_C[0]', H_C[0], (num_layers, batch_size, num_hidden_units))
            assert_shape('H_C[1]', H_C[1], (num_layers, batch_size, num_hidden_units))

            pred = self._vocab.decode_one_hot(outputs[0][0])
            sequence.append(pred)

        return ''.join(sequence[1:])

    def clip_gradients(self) -> None:
        total = torch.tensor([0.0], device=device)
        named_params = []
        for name, param in self.named_parameters(recurse=True):
            if not param.requires_grad:
                continue
            assert param.grad is not None, f"gradient is None for {name}"
            total += torch.sum(param.grad ** 2)
            named_params.append((name, param))

        norm = torch.sqrt(total)
        if norm <= self._grad_clip:
            return

        # Clip gradients
        clip_ratio = self._grad_clip / norm
        for name, param in named_params:
            param.grad *= clip_ratio
            logger.debug("{}'s gradient clipped, norm = {}, clip_ratio = {}",
                         name, norm, clip_ratio.item())

    def _forward(
        self,
        inputs: torch.Tensor,
        H_C: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters:
        - inputs: the input tensor with the shape of (batch_size, num_steps)
        - H_C: the tuple of hidden and cell states both with the shape of
            (num_layers, batch_size, num_hidden_units)

        Returns a tuple of two tensors:
        - item #0: the outputs with the shape of (batch_size, num_steps, vocab_size)
        - item #1: the tuple of hidden and cell states both with the shape of
            (num_layers, batch_size, num_hidden_units)
        """

        assert_dimensions('inputs', inputs, 2)

        X = self._vocab.one_hot_encode(inputs)

        outputs, H_C = self._mem_cell(X, H_C)
        assert isinstance(outputs, torch.Tensor)
        assert H_C is not None and isinstance(H_C, tuple)

        return self._output_layer(outputs), H_C


class Trainer:
    def __init__(
        self,
        model: LSTMLMConcise,
        dataset: TimeMachineDataset,
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
            y_indices = y.to(device)

            y_pred_logits = self._model(X)

            loss = self._loss_measurer(
                y_pred_logits.view(-1, y_pred_logits.shape[-1]),
                y_indices.view(-1))

            # Backpropagation
            loss.backward()
            self._model.clip_gradients()
            self._optimizer.step()
            self._optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


class Validator:
    def __init__(
        self,
        model: LSTMLMConcise,
        dataset: TimeMachineDataset,
        loss_measurer: nn.CrossEntropyLoss,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss_measurer = loss_measurer

    def __call__(self) -> tuple[float, float, float]:
        """
        Validates the model.

        Returns a tuple of three values:
        - item #0: the mean loss
        - item #1: the accuracy
        - item #2: the perplexity
        """
        num_batches = 0
        total_loss = 0.0
        num_correct = 0
        num_total = 0

        self._model.eval()
        with torch.inference_mode():
            for X, y in self._dataset.get_data_loader(train=False):
                X = X.to(device)
                y_indices = y.to(device) # shape is (batch_size, num_steps)

                y_pred_logits = self._model(X)

                total_loss += self._loss_measurer(
                    y_pred_logits.view(-1, y_pred_logits.shape[-1]),
                    y_indices.view(-1)
                    ).item()
                num_batches += 1

                y_pred_indices = self.index(y_pred_logits).flatten()
                num_correct += (y_pred_indices == y_indices.flatten()).sum().item()
                num_total += len(y_pred_indices)

        mean_loss = total_loss / num_batches
        accuracy = num_correct / num_total
        perplexity = torch.exp(torch.tensor(mean_loss)).item()
        return (mean_loss, accuracy, perplexity)

    def index(self, y: torch.Tensor) -> torch.Tensor:
        return y.argmax(dim=-1)


class MetricsPlotter:
    def __init__(
        self,
    ) -> None:
        self._epochs = []
        self._train_losses = []
        self._evaluate_losses = []
        self._perplexities = []

    def add(
        self,
        epoch: int,
        train_loss: float,
        validate_loss: float,
        perplexity: float,
    ) -> None:
        assert isinstance(train_loss, float), \
            f"Invalid type for train_loss (expected 'float', got {type(train_loss)})"
        assert isinstance(validate_loss, float), \
            f"Invalid type for evaluate_loss (expected 'float', got {type(validate_loss)})"
        assert isinstance(perplexity, float), \
            f"Invalid type for accuracy (expected 'float', got {type(perplexity)}"

        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._evaluate_losses.append(validate_loss)
        self._perplexities.append(perplexity)

    def plot(
        self,
        title: str,
        filename: str,
    ) -> None:
        fig, ax1 = plt.subplots()

        # Plot losses on the first y-axis
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss', color='tab:red')
        ax1.plot(self._epochs, self._train_losses, 'b', label='Train Loss')
        ax1.plot(self._epochs, self._evaluate_losses, 'r', label='Validation Loss')
        ax1.tick_params(axis='y', labelcolor='tab:red')
        ax1.legend(loc='upper left')

        # Create a second y-axis for perplexity
        ax2 = ax1.twinx()
        # Ensure ax2 is of the same type as ax1 (i.e., matplotlib.axes.Axes)
        # for linter type-checking
        assert isinstance(ax2, type(ax1))
        ax2.set_ylabel('perplexity', color='tab:blue')
        ax2.plot(
            self._epochs, self._perplexities, 'g', label='Validation Perplexity', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0)
        ax2.legend(loc='upper right')

        plt.title(title)
        plt.show()

        if filename is not None:
            fig.savefig(filename)


def preview(
    dataset: TimeMachineDataset,
    num: int = 4,
) -> None:
    if num <= 0:
        return

    logger.info("preview:")
    total = 0
    for X, Y in dataset.get_data_loader():
        for x, y in zip(X, Y):
            logger.info("  x = {}, y = {}", dataset.vocab.decode(x), dataset.vocab.decode(y))
            total += 1
            if total >= num:
                return


def main(
    num_preview: int = 4
) -> None:
    batch_size = 1024
    num_steps = 32
    num_hidden_units = 64
    max_epochs = 20
    learning_rate = 4.0
    weight_decay = 1e-5
    prefix = "it has"
    num_pred = 20
    start_sec = time.time()

    # Initialize dataset
    dataset = TimeMachineDataset(
        filepath="../timemachine.txt", batch_size=batch_size, num_steps=num_steps)
    preview(dataset, num_preview)

    # Initialize model, loss, and optimizer
    model = LSTMLMConcise(dataset.vocab, num_hidden_units)
    loss_measurer = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), learning_rate, weight_decay)

    # Train and validate the model
    trainer = Trainer(model, dataset, loss_measurer, optimizer)
    validator = Validator(model, dataset, loss_measurer)
    plotter = MetricsPlotter()

    logger.info("max_epochs = {}, batch_size = {}, learning_rate = {:.2f}, weight_decay = {:.1e}",
                max_epochs, batch_size, learning_rate, weight_decay)
    logger.info("num_hidden_units = {}, num_steps = {}, device = {}",
                num_hidden_units, num_steps, device)
    for epoch in range(max_epochs):
        train_loss = trainer.fit()
        validate_loss, accuracy, perplexity = validator()

        logger.info("epoch #{}, train_loss = {:.3f}, validate_loss = {:.3f}, "
                    "accuracy = {:.1%}, perplexity = {:.2f}",
                    epoch, train_loss, validate_loss, accuracy, perplexity)
        plotter.add(epoch, train_loss, validate_loss, perplexity)

    logger.info("device = {}, elapsed time: {:.1f} seconds", device, time.time() - start_sec)

    # Visualize samples (both correct and wrong predictions) from the last epoch
    plotter.plot(title=f"Language Model ({LSTMLMConcise.__name__})", filename="metrics.jpg")

    logger.info("prediction for '{}': '{}'", prefix, model.predict(prefix, num_pred))
    logger.info("done!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sink=sys.stderr, level="DEBUG")

    main()

    # Final output:
    # epoch #19, train_loss = 1.464, validate_loss = 1.793, accuracy = 47.3%, perplexity = 6.01
    # device = cuda, elapsed time: 26.8 seconds
    # prediction for 'it has': ' to me the strange t'
    # done!
