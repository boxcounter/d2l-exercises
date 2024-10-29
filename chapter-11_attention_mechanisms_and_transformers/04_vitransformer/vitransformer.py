# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments, too-many-instance-attributes
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals

import math
import time
import sys

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def assert_tensor(
    tensor_name: str,
    tensor,
) -> None:
    assert tensor is not None, f"{tensor_name} is None"
    assert isinstance(tensor, torch.Tensor), \
        f"Invalid type for {tensor_name} (expected 'torch.Tensor', got {type(tensor)})"


def assert_dimension(
    tensor_name: str,
    tensor,
    expected_dimensions: int | tuple[int, ...],
) -> None:
    assert_tensor(tensor_name, tensor)

    if isinstance(expected_dimensions, int):
        assert tensor.dim() == expected_dimensions, \
                (f"Invalid dimensions for {tensor_name} (expected {expected_dimensions}, "
                f"got {tensor.dim()})")
    elif isinstance(expected_dimensions, tuple):
        assert tensor.dim() in expected_dimensions, \
                (f"Invalid dimensions for {tensor_name} (expected one of {expected_dimensions}, "
                f"got {tensor.dim()})")
    else:
        raise ValueError(
            "Invalid type for 'expected_dimensions' (expected 'int' or 'tuple', "
            f"got {type(expected_dimensions)})")


def assert_dimension_size(
    tensor_name: str,
    tensor,
    dim: tuple[int,...] | int,
    size: tuple[int, ...] | int,
) -> None:
    assert_tensor(tensor_name, tensor)

    if isinstance(dim, int):
        assert isinstance(size, int), \
            ("Invalid type for 'expected_dimension_size' (expected 'int', "
             f"got {type(size)})")

        assert tensor.shape[dim] == size, \
            (f"Invalid size for dimension {dim} of {tensor_name} "
                f"(expected {size}, got {tensor.shape[dim]})")
    elif isinstance(dim, tuple):
        assert isinstance(size, tuple), \
            ("Invalid type for 'expected_dimension_size' (expected 'tuple', "
                f"got {type(size)})")

        assert len(dim) == len(size), \
            (f"Mismatched length between 'dimension' and 'expected_dimension_size' "
                f"({len(dim)} vs. {len(size)})")

        for d, size in zip(dim, size):
            assert tensor.size(d) == size, \
                (f"Invalid size for dimension {d} of {tensor_name} "
                    f"(expected {size}, got {tensor.size(d)})")
    else:
        raise ValueError(
            f"Invalid type for 'dimension' (expected 'int' or 'tuple', got {type(dim)})")


def assert_same_shape(
    tensor1_name: str,
    tensor1,
    tensor2_name: str,
    tensor2,
) -> None:
    assert_tensor(tensor1_name, tensor1)
    assert_tensor(tensor2_name, tensor2)
    assert tensor1.shape == tensor2.shape, \
        (f"Mismatched shape between {tensor1_name} and {tensor2_name} "
         f"({tensor1.shape} vs. {tensor2.shape})")


def assert_same_partial_shape(
    tensor1_name: str,
    tensor1,
    tensor2_name: str,
    tensor2,
    *,
    dim: tuple[int, ...] | int,
) -> None:
    """
    Asserts that the two tensors have the same shape at the specified dimensions.
    """

    assert_tensor(tensor1_name, tensor1)
    assert_tensor(tensor2_name, tensor2)

    if isinstance(dim, int):
        dim = (dim, )

    for d in dim:
        assert tensor1.shape[d] == tensor2.shape[d], \
            (f"Mismatched shape at dimension {d} between {tensor1_name} and {tensor2_name} "
             f"({tensor1.shape[d]} vs. {tensor2.shape[d]})")


def assert_shape(
    tensor_name: str,
    tensor,
    expected_shape: tuple[int, ...],
) -> None:
    assert_tensor(tensor_name, tensor)
    assert tensor.shape == expected_shape, \
        f"Invalid shape for {tensor_name} (expected {expected_shape}, got {tensor.shape})"


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
                                 gridspec_kw={'hspace': row_spacing},
                                 squeeze=False)  # Ensure axes is always a 2D array

        axes = axes.flatten()  # Now this should work without error
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
            labels = ['Real: ' + label + '\n' + 'Pred: ' + plabel
                      for label, plabel in zip(self.text_labels(y), self.text_labels(y_pred))]
        elif len(batch) == 2:
            X, y = batch
            labels = self.text_labels(y)
        else:
            logger.error("Invalid batch length (expected 2 or 3, got {})", len(batch))
            return

        X = X.squeeze(1)  # Remove channel dimension for grayscale image
        nrows = math.ceil(len(X) / ncols)
        self.show_images(X, nrows, ncols, titles=labels, save_filename=filename)


def _init_weights_fn(module: nn.Module) -> None:
    if not isinstance(module, (nn.Embedding, nn.GRU)):
        return

    for name, param in module.named_parameters(recurse=False):
        if len(param.shape) < 2:
            # Skip bias, otherwise the later xavier initialization will trigger an
            # ValueError("Fan in and fan out can not be computed for tensor with
            # fewer than 2 dimensions")
            continue

        if "weight" in name:
            logger.debug("initialize weight '{}' of module '{}", name, module)
        else:
            logger.warning("Initialize parameter '{}' of module '{}", name, module)

        nn.init.xavier_normal_(param)


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        width: int,
        height: int,
        num_hiddens: int
    ) -> None:
        super().__init__()

        patch_size = (width, height)
        self._net = nn.LazyConv2d(out_channels=num_hiddens,
                                  kernel_size=patch_size,
                                  stride=patch_size,
                                  device=device)

        self._width = width
        self._height = height
        self._num_hiddens = num_hiddens

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        assert_dimension('X', X, 4)

        batch_size, _, image_width, image_height = X.size()
        patch_width, patch_height = self._width, self._height

        num_patches_horizontal = image_width // patch_width
        num_patches_vertical = image_height // patch_height

        # Convolution
        outputs = self._net(X)
        assert_shape('outputs', outputs,
                     (batch_size, self._num_hiddens, num_patches_horizontal, num_patches_vertical))

        # Flatten
        outputs = outputs.flatten(start_dim=2) # The final two dimensions are flattened
        num_patches = num_patches_horizontal * num_patches_vertical
        assert_shape('outputs', outputs,
                     (batch_size, self._num_hiddens, num_patches))

        # Transpose
        outputs = outputs.transpose(1, 2)
        assert_shape('outputs', outputs, (batch_size, num_patches, self._num_hiddens))

        return outputs


def masked_softmax(
    attention_scores: torch.Tensor,
    valid_lens: torch.Tensor | None,
) -> torch.Tensor:
    """
    Computes the softmax of the attention scores after masking the padding elements.

    Parameters:
    - attention_scores: the 3D tensor of attention scores with the shape of
        (batch_size, num_queries, num_keys)
    - valid_lens: a tensor that handles different usecases:
        - If it's a 1D tensor, it specifies the valid lengths for the keys,
          used in the encoder's self-attention and the decoder's cross-attention.
        - If it's a 2D tensor, it specifies the valid lengths for the queries,
          used in the decoder's self-attention.
        - If it's None, indicates it's during prediction.

        Note: While including usecase details might not adhere to the best practice of
              software engineering, it helps provide a better understanding of the code
              for readers including myself.

    Returns the softmax of the attention scores.
    """

    assert_dimension('attention_scores', attention_scores, 3)

    if valid_lens is None:
        return torch.softmax(attention_scores, dim=-1)

    assert_dimension('valid_lens', valid_lens, (1, 2))
    # Ensure the same batch size
    assert_same_partial_shape(
        'attention_scores', attention_scores, 'valid_lens', valid_lens, dim=0)

    batch_size, num_queries, num_keys = attention_scores.shape

    if valid_lens.dim() == 1:
        valid_lens = valid_lens.reshape(-1, 1, 1).repeat(1, num_queries, num_keys)
    else:
        assert_shape('valid_lens', valid_lens, (batch_size, num_queries))
        valid_lens = valid_lens.reshape(batch_size, num_queries, 1).repeat(1, 1, num_keys)
    assert_shape('valid_lens', valid_lens, (batch_size, num_queries, num_keys))

    mask = torch.arange(0, num_keys, device=device).repeat(batch_size, num_queries, 1)
    assert_shape('mask', mask, (batch_size, num_queries, num_keys))
    mask = mask >= valid_lens # True for the padding elements

    # Mask off the padding elements
    attention_scores = attention_scores.masked_fill(mask, -torch.inf)

    return torch.softmax(attention_scores, dim=-1)


class PositionalEmbedding(nn.Module):
    def __init__(
        self,
        num_hiddens: int,
    ) -> None:
        super().__init__()

        self._pos = nn.Parameter(
            torch.zeros((1, 1, num_hiddens), dtype=torch.float32, device=device)
        )

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        assert_dimension('X', X, 3)
        outputs = X + self._pos # utilize the broadcasting mechanism
        assert_same_shape('outputs', outputs, 'X', X)
        return outputs


class DotProductAttention(nn.Module):
    def __init__(
        self,
        dropout: float,
    ) -> None:
        super().__init__()
        self._dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - queries: the queries with the shape of (batch_size, num_queries, num_hidden_units)
        - keys: the keys with the shape of (batch_size, num_keys, num_hidden_units)
        - values: the values with the shape of (batch_size, num_keys, num_features_of_values)
        - valid_lens: a tensor that handles different usecases:
            - If it's a 1D tensor, it specifies the valid lengths for the keys,
              used in the encoder's self-attention and the decoder's cross-attention.
            - If it's a 2D tensor, it specifies the valid lengths for the queries,
              used in the decoder's self-attention.
            - If it's None, indicates it's during prediction.

            Note: While including usecase details might not adhere to the best practice of
                  software engineering, it helps provide a better understanding of the code
                  for readers including myself.

        Returns a tuple of two tensors:
        - the attention (context) of shape (batch_size, num_queries, num_features_of_values).
        - the attention weights of shape (batch_size, num_queries, num_keys).
        """

        assert_dimension('queries', queries, 3)
        assert_dimension('keys', keys, 3)
        assert_dimension('values', values, 3)
        assert_same_shape('keys', keys, 'values', values)
        # Ensure the same number of hidden units
        assert_same_partial_shape('queries', queries, 'keys', keys, dim=2)

        if valid_lens is not None:
            assert_dimension('valid_lens', valid_lens, (1, 2))

            if valid_lens.dim() == 1:
                assert_same_partial_shape('valid_lens', valid_lens, 'keys', keys, dim=0)
            else:
                assert_same_shape('quries', queries, 'keys', keys)
                assert_same_partial_shape('valid_lens', valid_lens, 'queries', queries, dim=(0, 1))

        batch_size, num_queries, num_hidden_units = queries.shape
        num_keys = keys.size(1)

        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(num_hidden_units)
        assert_shape('scores', scores, (batch_size, num_queries, num_keys))

        weights = masked_softmax(scores, valid_lens)
        assert_shape('weights', weights, (batch_size, num_queries, num_keys))

        assert_shape('values', values, (batch_size, num_keys, num_hidden_units))
        context = torch.bmm(self._dropout(weights), values)
        assert_shape('context', context, (batch_size, num_queries, num_hidden_units))

        return (context, weights)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        num_hidden_units: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self._W_q = nn.LazyLinear(num_hidden_units, bias=False, device=device)
        self._W_k = nn.LazyLinear(num_hidden_units, bias=False, device=device)
        self._W_v = nn.LazyLinear(num_hidden_units, bias=False, device=device)
        self._W_o = nn.LazyLinear(num_hidden_units, bias=False, device=device)
        self._attention = DotProductAttention(dropout)

        self._num_hidden_units = num_hidden_units
        self._num_heads = num_heads

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - queries: the queries with the shape of (batch_size, num_queries, num_hidden_units)
        - keys: the keys with the shape of (batch_size, num_keys, num_hidden_units)
        - values: the values with the shape of (batch_size, num_values, num_hidden_units)
        - valid_lens: the valid lengths of the keys with the shape of (batch_size,)

        Returns the multi-head attention with the shape of
            (batch_size, num_queries, num_hidden_units).
        """

        assert_dimension('queries', queries, 3)
        assert_dimension('keys', keys, 3)
        assert_dimension('values', values, 3)
        # Ensure the same batch size
        assert_same_partial_shape('queries', queries, 'keys', keys, dim=0)
        assert_same_partial_shape('keys', keys, 'values', values, dim=0)

        if valid_lens is not None:
            assert_dimension('valid_lens', valid_lens, (1, 2))
            assert_same_partial_shape('values', values, 'valid_lens', valid_lens, dim=0)

        num_heads, hidden_size = self._num_heads, self._num_hidden_units

        q, k, v = self._W_q(queries), self._W_k(keys), self._W_v(values)
        assert_shape('q', q, (queries.size(0), queries.size(1), hidden_size))
        assert_shape('k', k, (keys.size(0), keys.size(1), hidden_size))
        assert_shape('v', v, (values.size(0), values.size(1), hidden_size))

        hidden_size_per_head = hidden_size // num_heads
        q, k, v = self._split_qkv_into_multi_head(q, k, v)
        assert_shape('q', q, (queries.size(0) * num_heads, queries.size(1), hidden_size_per_head))
        assert_shape('k', k, (keys.size(0) * num_heads, keys.size(1), hidden_size_per_head))
        assert_shape('v', v, (values.size(0) * num_heads, values.size(1), hidden_size_per_head))

        if valid_lens is not None:
            valid_lens = valid_lens.repeat_interleave(num_heads, dim=0)

        context, weights = self._attention(q, k, v, valid_lens)
        assert_shape('context', context,
                     (queries.size(0) * num_heads, queries.size(1), hidden_size_per_head))

        context = self._W_o(self._merge_multi_head_outputs(context))
        assert_shape('context', context, (queries.size(0), queries.size(1), hidden_size))

        return context, weights

    def _split_qkv_into_multi_head(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Splits the queries, keys, and values into multiple heads.

        Parameters:
        - queries: the queries with the shape of (batch_size, num_queries, num_hidden_units)
        - keys: the keys with the shape of (batch_size, num_keys, num_hidden_units)
        - values: the values with the shape of (batch_size, num_values, num_hidden_units)

        Returns the queries, keys, and values with the shape of
            (batch_size * num_heads, num_queries, num_hidden_units // num_heads).
        """

        assert_dimension('queries', queries, 3)
        assert_dimension('keys', keys, 3)
        assert_dimension('values', values, 3)
        assert_same_shape('keys', keys, 'values', values)

        hidden_size_per_head = self._num_hidden_units // self._num_heads

        tensors = {
            'queries': queries,
            'keys': keys,
            'values': values,
        }
        for name, t in tensors.items():
            batch_size, seq_len, num_hidden_units = t.shape
            assert num_hidden_units == self._num_hidden_units, \
                (f"Mismatched number of hidden units at dimension 2 between "
                 f"tensors (expected {self._num_hidden_units}, got {num_hidden_units})")

            t = t.reshape(batch_size, seq_len, self._num_heads, hidden_size_per_head)
            t = t.permute(0, 2, 1, 3)
            assert_shape(f'{name}', t,
                         (batch_size, self._num_heads, seq_len, hidden_size_per_head))
            tensors[name] = t.reshape(
                batch_size * self._num_heads, seq_len, hidden_size_per_head)

        return (tensors['queries'], tensors['keys'], tensors['values'])

    def _merge_multi_head_outputs(
        self,
        multi_head_outputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Merges the output of multiple heads into a single tensor.

        Parameters:
        - multi_head_outputs: the output of multiple heads with the shape of
            (batch_size * num_heads, num_queries, num_hidden_units // num_heads)

        Returns the merged output with the shape of (batch_size, num_queries, num_hidden_units).
        """

        assert_dimension('multi_heads_output', multi_head_outputs, 3)
        extended_batch_size, seq_len, hidden_size_per_head = multi_head_outputs.shape
        assert extended_batch_size % self._num_heads == 0

        batch_size = extended_batch_size // self._num_heads
        outputs = multi_head_outputs.reshape(
            batch_size, self._num_heads, seq_len, hidden_size_per_head)
        outputs = outputs.permute(0, 2, 1, 3)
        assert_shape('outputs', outputs,
                     (batch_size, seq_len, self._num_heads, hidden_size_per_head))
        outputs = outputs.reshape(batch_size, seq_len, -1)

        return outputs


class MLP(nn.Module):
    def __init__(
        self,
        num_hiddens: int,
        num_outputs: int,
        dropout: float,
    ) -> None:
        """
        Parameters:
        - num_hiddens: the number of hidden units in the MLP.
        - num_outputs: the number of outputs of the MLP.
        - dropout: the dropout rate.
        """

        super().__init__()

        self._net = nn.Sequential(
            nn.LazyLinear(num_hiddens, device=device),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LazyLinear(num_outputs, device=device),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        X: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters:
        - X: the input tensor with the shape of (batch_size, num_inputs, num_hiddens)

        Returns the output tensor with the shape of (batch_size, num_outputs).
        """

        assert_dimension('X', X, 3)
        return self._net(X)


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_mlp_hiddens: int,
        num_heads: int,
        num_hiddens: int,
        dropout: float,
    ) -> None:
        """
        Parameters:
        - num_mlp_hiddens: the number of hidden units in the MLP.
        - num_heads: the number of heads in the multi-head attention.
        - num_hiddens: the number of hidden units in the multi-head attention.
        - dropout: the dropout rate.
        """

        super().__init__()

        self._norm1 = nn.LayerNorm(num_hiddens, device=device)
        self._mha = MultiHeadAttention(num_hiddens, num_heads, dropout)
        self._norm2 = nn.LayerNorm(num_hiddens, device=device)
        self._mlp = MLP(num_hiddens=num_mlp_hiddens, num_outputs=num_hiddens, dropout=dropout)

        self._num_hiddens = num_hiddens

        # For visualization
        self.attention_weights = torch.tensor([])

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - X: the input tensor with the shape of (batch_size, num_inputs, num_hiddens)

        Returns the output tensor with the shape of (batch_size, num_inputs, num_hiddens).
        """

        assert_dimension('X', X, 3)
        assert_dimension_size('X', X, 2, self._num_hiddens)

        shape = X.shape

        # Norm and Multi-head attention
        inputs = X
        outputs = self._norm1(inputs)
        assert_shape('outputs', outputs, shape)
        queries = keys = values = outputs
        outputs, weights = self._mha(queries, keys, values, valid_lens=None)
        assert_shape('outputs', outputs, shape)
        outputs = inputs + outputs
        self.attention_weights = weights

        # Norm and MLP
        inputs = outputs
        outputs = self._norm2(inputs)
        assert_shape('outputs', outputs, shape)
        outputs = self._mlp(outputs)
        assert_shape('outputs', outputs, shape)
        outputs = inputs + outputs
        assert_shape('outputs', outputs, shape)

        return outputs


class ViTransformer(nn.Module):
    def __init__(
        self,
        patch_width: int,
        path_height: int,
        num_blocks: int,
        num_mlp_hiddens: int,
        num_heads: int,
        num_hiddens: int,
        num_labels: int,
        grad_clip_threshold: float,
        dropout: float,
    ) -> None:
        """
        Parameters:
        - patch_width: the width of the patch.
        - path_height: the height of the patch.
        - num_blocks: the number of encoder blocks.
        - num_mlp_hiddens: the number of hidden units in the MLP.
        - num_heads: the number of heads in the multi-head attention.
        - num_hiddens: the number of hidden units in the multi-head attention.
        - num_labels: the number of labels.
        - grad_clip_threshold: the threshold for gradient clipping.
        - dropout: the dropout rate.
        """

        super().__init__()

        self._embedding = PatchEmbedding(patch_width, path_height, num_hiddens)
        self._positional_embedding = PositionalEmbedding(num_hiddens)
        self._dropout = nn.Dropout(dropout)
        self._blocks = nn.Sequential(
            *[EncoderBlock(num_mlp_hiddens, num_heads, num_hiddens, dropout)
              for _ in range(num_blocks)])
        self._head = nn.Sequential(
            nn.LayerNorm(num_hiddens, device=device),
            nn.LazyLinear(num_labels, device=device)
        )
        self._cls_token = nn.Parameter(torch.zeros(num_hiddens, device=device))

        self._num_hiddens = num_hiddens
        self._num_labels = num_labels
        self._grad_clip_threshold = grad_clip_threshold

        self.apply(_init_weights_fn)

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - X: the input tensor with the shape of (batch_size, num_patches, patch_width, patch_height)

        Returns the output tensor with the shape of (batch_size, num_labels).
        """

        # Apply patch embedding
        assert_dimension('X', X, 4)
        inputs = self._embedding(X)
        assert_dimension('inputs', inputs, 3)
        assert_dimension_size('inputs', inputs, 2, self._num_hiddens)
        batch_size, num_steps, num_hiddens = inputs.shape

        # Cat the <cls> token
        cls_tokens = self._cls_token.view(1, 1, -1).expand(batch_size, -1, -1)
        inputs = torch.cat((cls_tokens, inputs), dim=1)
        assert_shape('inputs', inputs, (batch_size, num_steps + 1, num_hiddens))

        # Apply positional embedding
        inputs = self._positional_embedding(inputs)
        assert_shape('inputs', inputs, (batch_size, num_steps + 1, num_hiddens))

        # Dropout
        inputs = self._dropout(inputs)

        # Encoder blocks
        outputs = self._blocks(inputs)
        assert_shape('outputs', outputs, (batch_size, num_steps + 1, self._num_hiddens))

        # Only the representation of <cls> is used in the output layers.
        outputs = self._head(outputs[:, 0])
        assert_shape('outputs', outputs, (batch_size, self._num_labels))

        return outputs

    @property
    def attention_weights(self) -> torch.Tensor:
        weights = []
        for block in self._blocks:
            weights.append(block.attention_weights)

        return torch.stack(weights)

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
        if norm <= self._grad_clip_threshold:
            return

        # Clip gradients
        clip_ratio = self._grad_clip_threshold / norm
        for name, param in named_params:
            param.grad *= clip_ratio

        logger.trace("gradients clipped, norm = {:.2f}, clip_ratio = {:.2f}",
                     norm.item(), clip_ratio.item())


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


class Trainer:
    def __init__(
        self,
        model: ViTransformer,
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
            self._model.clip_gradients()
            self._optimizer.step()
            self._optimizer.zero_grad()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / num_batches


class Validator:
    def __init__(
        self,
        model: ViTransformer,
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

    def index(self, y: torch.Tensor) -> torch.Tensor:
        return y.argmax(dim=1)

class MetricsPlotter:
    def __init__(
        self,
    ) -> None:
        self._epochs = []
        self._train_losses = []
        self._evaluate_losses = []
        self._accuracy = []

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
        self._accuracy.append(accuracy)

    def plot(
        self,
        title: str,
        filename: str = "metrics.jpg",
    ) -> None:
        """
        Plots the training and validation loss, and the accuracy.

        Thanks to Claude Sonnet 3.5 for sparing me the matplotlib wrestling match!
        """

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
        ax2.set_ylabel('accuracy', color='tab:blue')
        ax2.plot(
            self._epochs, self._accuracy, 'g', label='Validation accuracy', linestyle='--')
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        ax2.set_ylim(0)
        ax2.legend(loc='upper right')

        plt.title(title)
        plt.show()

        if filename is not None:
            fig.savefig(filename)


class AttentionWeigthsVisualizer:
    def __init__(self) -> None:
        self._epochs = []
        self._encoder_attention_weights = []
        self._decoder_causal_attention_weights = []
        self._decoder_cross_attention_weights = []

    def add(
        self,
        epoch: int,
        encoder_attention_weights: torch.Tensor,
        decoder_causal_attention_weights: torch.Tensor,
        decocer_cross_attention_weights: torch.Tensor,
    ) -> None:
        self._epochs.append(epoch)
        self._encoder_attention_weights.append(encoder_attention_weights)
        self._decoder_causal_attention_weights.append(decoder_causal_attention_weights)
        self._decoder_cross_attention_weights.append(decocer_cross_attention_weights)

    def __call__(
        self,
        start: int = 0,
    ) -> None:
        start = max(0, start)
        end = len(self._epochs)

        for i in range(start, end):
            epoch = self._epochs[i]
            encoder_attention_weights = self._encoder_attention_weights[i]
            decoder_causal_attention_weights = self._decoder_causal_attention_weights[i]
            decoder_cross_attention_weights = self._decoder_cross_attention_weights[i]

            title = f"Encoder attention weights (epoch #{epoch})"
            filename = f"encoder_attention_weights_epoch_{epoch}.jpg"
            self._visualize(encoder_attention_weights, title, filename)

            title = f"Decoder causal-attention weights (epoch #{epoch})"
            filename = f"decoder_causal_attention_weights_epoch_{epoch}.jpg"
            self._visualize(decoder_causal_attention_weights, title, filename)

            title = f"Decoder cross-attention weights (epoch #{epoch})"
            filename = f"decoder_cross_attention_weights_epoch_{epoch}.jpg"
            self._visualize(decoder_cross_attention_weights, title, filename)

    def _visualize(
        self,
        attention_weights: torch.Tensor,
        title: str,
        filename: str,
        cmap: str = 'Reds',
    ) -> None:
        """
        Plots the heatmap for the entire attention weights tensor as a heat grid.

        Thanks to Claude Sonnet 3.5 for sparing me the matplotlib wrestling match!

        Parameters:
        - attention_weights: tensor of shape (num_blocks, num_heads, num_queries, num_keys)
        - title: title of the heatmap
        - filename: name of the file to save the plot
        - cmap: colormap used for the heatmap
        """

        attention_weights = attention_weights.detach().cpu().numpy()
        num_blocks, num_heads, num_queries, num_keys = attention_weights.shape

        plt.close('all')  # Close all existing figures
        fig, axes = plt.subplots(
            num_blocks,
            num_heads + 1,
            figsize=(3 * (num_heads + 1), 3 * num_blocks),
            squeeze=False,
            gridspec_kw={'width_ratios': [0.2] + [1] * num_heads})
        fig.suptitle(title, fontsize=16)

        im = None
        for block in range(num_blocks):
            axes[block, 0].text(
                0.5, 0.5, f'Block {block}', rotation=90,
                verticalalignment='center', horizontalalignment='center', fontsize=10)
            axes[block, 0].axis('off')  # Turn off axis for the block label

            for head in range(num_heads):
                ax = axes[block, head + 1]  # Shift the head plots one column to the right
                im = ax.imshow(
                    attention_weights[block, head],
                    cmap=cmap,
                    aspect='equal',
                    vmin=0,
                    vmax=1,
                )

                ax.set_title(f'Head {head}', fontsize=8)
                ax.set_xlabel('Keys', fontsize=8)
                ax.set_ylabel('Queries', fontsize=8)

                # Set tick labels
                ax.set_xticks(range(num_keys))
                ax.set_yticks(range(num_queries))
                ax.set_xticklabels(range(num_keys), fontsize=6)
                ax.set_yticklabels(range(num_queries), fontsize=6)

                # Add grid lines
                ax.set_xticks(np.arange(num_keys+1)-.5, minor=True)
                ax.set_yticks(np.arange(num_queries+1)-.5, minor=True)
                ax.grid(which="minor", color="w", linestyle='-', linewidth=0.5)
                ax.tick_params(which="minor", bottom=False, left=False)

        # Add a single colorbar to the right of the entire figure
        fig.subplots_adjust(right=0.92, top=0.9, bottom=0.1, left=0.05, hspace=0.3, wspace=0.3)
        if im is not None:
            cbar_ax = fig.add_axes((0.94, 0.15, 0.02, 0.7))
            fig.colorbar(im, cax=cbar_ax)

        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.show()
        plt.close(fig)  # Close the figure after showing and saving


def main(
    preview_dataset: bool = True
) -> None:
    max_epochs = 10
    num_samples = 16
    learning_rate = 0.15
    batch_size = 128
    input_shape = (1, 96, 96)
    num_blocks = 2
    num_heads = 8
    num_hiddens = 512
    num_mlp_hiddens = 2048
    patch_width = 16
    patch_height = 16
    weight_decay = 1e-5
    grad_clip_threshold = 1.0
    dropout = 0.1
    start_sec = time.time()

    # Print hyperparameters
    logger.info("max_epochs = {}, learning_rate = {:.3f}, batch_size = {}, device = {}",
                max_epochs, learning_rate, batch_size, device)
    logger.info("weight_decay = {:.1e}, dropout = {:.2f}", weight_decay, dropout)
    logger.info("num_blocks = {}, num_heads = {}, num_hidden_units = {}",
                num_blocks, num_heads, num_hiddens)

    # Initialize dataset
    dataset = FashionMNISTDataset(batch_size=batch_size, resize=input_shape[1:])
    if preview_dataset:
        batch = next(iter(dataset.get_data_loader(False)))
        dataset.visualize(batch)

    # Initialize model, loss_measurer, and optimizer
    model = ViTransformer(
        patch_width, patch_height, num_blocks, num_mlp_hiddens, num_heads, num_hiddens,
        dataset.num_labels, grad_clip_threshold, dropout)
    loss_measurer = nn.CrossEntropyLoss()
    optimizer = optim.SGD(params=model.parameters(),
                          lr=learning_rate,
                          weight_decay=weight_decay)

    # Initialize trainer and validator
    trainer = Trainer(model, dataset, loss_measurer, optimizer)
    validator = Validator(model, dataset, loss_measurer)

    # Initialize plotter
    plotter = MetricsPlotter()

    # Train and validate
    samples: list[torch.Tensor] = []
    for epoch in range(max_epochs):
        train_loss = trainer.fit()
        validate_loss, accuracy, samples = validator(
            num_samples if epoch == max_epochs - 1 else 0)

        plotter.add(epoch, train_loss, validate_loss, accuracy)

        logger.info("epoch #{}, train_loss = {:.3f}, validate_loss = {:.3f}, accuracy = {:.1%} ",
                    epoch, train_loss, validate_loss, accuracy)

    logger.info("device = {}, elapsed time: {:.1f} seconds", device, time.time() - start_sec)

    # Visualize samples (both correct and wrong predictions) from the last epoch
    dataset.visualize(samples, filename="pred_samples.jpg")

    # Plot metrics
    plotter.plot(title=f"Machine Translation ({ViTransformer.__name__})")

    logger.info("done!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sink=sys.stderr, level="DEBUG")

    main(True)

    # Final output:
    # epoch #9, train_loss = 0.384, validate_loss = 0.462, accuracy = 84.0%
    # device = cuda, elapsed time: 113.2 seconds
    # done!
