# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments, too-many-instance-attributes
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals

import io
import sys
import re
import time
import math
import random
import zipfile
from collections import Counter
from typing import Final
import reprlib

import torch
import torch.utils
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from loguru import logger
import requests
import numpy as np


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


class Vocabulary:
    UNK: Final = '<unk>'
    PAD: Final = '<pad>'
    BOS: Final = '<bos>'
    EOS: Final = '<eos>'
    RESERVED_TOKENS: Final = {UNK, PAD, BOS, EOS}

    def __init__(
        self,
        lines_of_words: list[list[str]],
        min_freq: int = 0,
    ) -> None:
        self._word_token_mapping: dict[str, int] = {}
        self._words: list[str] = []

        words = self._extract_valid_words(lines_of_words, min_freq)
        self._words = sorted(words | self.RESERVED_TOKENS)
        self._word_token_mapping = {word: i for i, word in enumerate(self._words)}

    @property
    def size(self) -> int:
        return len(self._words)

    @property
    def pad_token(self) -> int:
        return self._tokenize_single(self.PAD)

    @property
    def eos_token(self) -> int:
        return self._tokenize_single(self.EOS)

    @property
    def bos_token(self) -> int:
        return self._tokenize_single(self.BOS)

    def tokenize(
        self,
        words: str | list[str],
    ) -> int | list[int]:
        if isinstance(words, str):
            assert len(words) == 1
            return self._tokenize_single(words)
        elif isinstance(words, list):
            return [self._tokenize_single(word) for word in words]
        else:
            raise ValueError(
                f"Invalid type for 'words', expected 'str' or 'list[str]', got {type(words)}")

    def untokenize(
        self,
        tokens: int | list[int],
    ) -> str | list[str]:
        if isinstance(tokens, int):
            return self._untokenize_single(tokens)
        elif isinstance(tokens, list):
            return [self._untokenize_single(token) for token in tokens]
        else:
            raise ValueError(
                f"Invalid type for 'tokens', expected 'int' or 'list[int]', got {type(tokens)}")

    def __str__(self) -> str:
        return f"{type(self).__name__}: size={self.size:,}, tokens={reprlib.repr(self._words)}"

    @staticmethod
    def _extract_valid_words(
        lines_of_words: list[list[str]],
        min_freq: int,
    ) -> set[str]:
        counter = Counter([word for line in lines_of_words for word in line])
        if min_freq in (0, 1):
            return set(counter.keys())

        return {word for word, count in counter.items() if count >= min_freq}

    def _tokenize_single(
        self,
        word: str
    ) -> int:
        assert isinstance(word, str)
        return self._word_token_mapping.get(
                word, self._word_token_mapping[self.UNK])

    def _untokenize_single(
        self,
        token: int
    ) -> str:
        assert isinstance(token, int)
        return self._words[token]


class TextSequenceDataset:
    URL: Final = "http://www.manythings.org/anki/cmn-eng.zip"

    def __init__(
        self,
        filepath: str,
        batch_size: int,
        training_set_ratio: float = 0.8,
        num_evaluation_sample: int = 1,
    ) -> None:
        try:
            with open(filepath, 'r', encoding='UTF-8') as f:
                content = f.read()

            if not content or len(content) == 0:
                raise Exception("Empty file")
        except Exception:
            logger.info("downloading content from {}", self.URL)
            content = self._download(self.URL, filepath)

        logger.debug("content length = {:,}", len(content))

        source_lines_of_words, target_lines_of_words = self._extract_lines_of_words(filepath)
        source_vocab = Vocabulary(source_lines_of_words)
        target_vocab = Vocabulary(target_lines_of_words)

        self._batch_size = batch_size
        self._source_vocab = source_vocab
        self._target_vocab = target_vocab
        self._training_set, self._validation_set = self._construct_dataset(
            *self.tokenize(source_lines_of_words, source_vocab),
            *self.tokenize(target_lines_of_words, target_vocab, insert_bos=True),
            training_set_ratio)
        self._evaluation_samples = self._extract_evaluation_samples(
            source_lines_of_words, target_lines_of_words, num_evaluation_sample)

    @property
    def source_vocab(self) -> Vocabulary:
        return self._source_vocab

    @property
    def target_vocab(self) -> Vocabulary:
        return self._target_vocab

    @property
    def evaluation_samples(self) -> list[tuple[str, str]]:
        return self._evaluation_samples

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

    @staticmethod
    def tokenize(
        lines_of_words: list[list[str]],
        vocab: Vocabulary,
        insert_bos: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenizes the lines of words with padding if necessary using the vocabulary.

        Parameters:
        - lines_of_words: a list of lines of words. For example:
            [
                ['this', 'is', 'a', 'sentence'],
                ['another', 'sentence']
            ]
        - vocab: the vocabulary.
        - insert_bos: whether to insert the <bos> token.

        Returns a tensor of two tensors:
        - tensor #1: the tokenized lines of words with the shape (batch_size, sequence_len)
            in which the sequence_len represents the number of words, also usually named as
            num_steps.
        - tensor #2: the valid length of each line of words with the shape (batch_size).
        """

        assert isinstance(lines_of_words, list)

        max_len = max(len(text) for text in lines_of_words) + (2 if insert_bos else 1)
        lines_of_tokens = []
        lines_of_valid_len = []

        for line in lines_of_words:
            # tokenize
            line = vocab.tokenize(line)
            assert isinstance(line, list)
            # bos
            if insert_bos:
                line.insert(0, vocab.bos_token)
            # eos
            line.append(vocab.eos_token)
            # pad
            num_padding = max_len - len(line)
            line.extend([vocab.pad_token] * num_padding)

            lines_of_tokens.append(line)
            lines_of_valid_len.append(max_len - num_padding)

        return (torch.tensor(lines_of_tokens, device=device),
                torch.tensor(lines_of_valid_len, device=device))

    @staticmethod
    def untokenize(
        tokens: torch.Tensor,
        vocab: Vocabulary,
    ) -> list[list[str]]:
        """
        Untokenizes the tokens using the vocabulary.

        Parameters:
        - tokens: a tensor of shape (batch_size, sequence_len).
        - vocab: the vocabulary.

        Returns a list of lines of words.
        """

        assert_dimension('tokens', tokens, 2)
        tokens = tokens.to(dtype=torch.int)

        lines_of_words = []
        for t in tokens: # iterate through the batch
            lines_of_words.append(vocab.untokenize(t.tolist()))

        return lines_of_words

    @staticmethod
    def preprocess_source_text_sequence(
        text_sequence: str
    ) -> list[str]:
        """
        Preprocesses the English text sequence by:
        - Lowercasing the text.
        - Replacing non-breaking spaces with spaces.
        - Inserting a space before the punctuation marks unless there is already one.

        Parameters:
        - text_sequence: the English text sequence.

        Returns a list of words.
        """

        # Lowercase and replace non-breaking spaces with space
        text = text_sequence.lower().replace('\u202f', ' ').replace('\xa0', ' ')

        # Insert a space before the punctuation marks unless there is already one.
        pattern = r'(?<!\s)([.,!?])'
        text = re.sub(pattern, r' \1', text)

        return text.split()

    @staticmethod
    def preprocess_target_text_sequence(
        text_sequence: str
    ) -> list[str]:
        """
        Preprocesses the Chinese text sequence.

        Parameters:
        - text_sequence: the Chinese text sequence.

        Returns a list of words.
        """

        return list(text_sequence)

    @staticmethod
    def _construct_dataset(
        source: torch.Tensor,
        source_valid_len: torch.Tensor,
        target: torch.Tensor,
        target_valid_len: torch.Tensor,
        training_set_ratio: float,
    ) -> tuple[TensorDataset, TensorDataset]:
        assert_same_partial_shape(
            'source', source, 'target', target, dim=0)
        assert_same_partial_shape(
            'source', source, 'source_valid_len', source_valid_len, dim=0)
        assert_same_partial_shape(
            'source_valid_len', source_valid_len, 'target_valid_len', target_valid_len, dim=0)

        training_set_stop = int(source.shape[0] * training_set_ratio)

        # Inputs for encoder
        source_train = source[:training_set_stop]
        source_valid_len_train = source_valid_len[:training_set_stop]
        source_validation = source[training_set_stop:]
        source_valid_len_validation = source_valid_len[training_set_stop:]

        # Inputs and labels for decoder
        # The labels are the orginal target shifted by one token
        target_X_train = target[:training_set_stop, :-1]
        target_X_valid_len_train = target_valid_len[:training_set_stop] - 1
        target_Y_train = target[:training_set_stop, 1:]
        target_X_validation = target[training_set_stop:, :-1]
        target_X_valid_len_validation = target_valid_len[training_set_stop:] - 1
        target_Y_validation = target[training_set_stop:, 1:]

        return (TensorDataset(source_train, source_valid_len_train,
                              target_X_train, target_X_valid_len_train,
                              target_Y_train),
                TensorDataset(source_validation, source_valid_len_validation,
                              target_X_validation, target_X_valid_len_validation,
                              target_Y_validation))

    @staticmethod
    def _download(
        url: str,
        filepath: str,
    ) -> str:
        """
        Downloads the content from the URL and writes it to the file.
        """

        headers = {
            'User-Agent': 'd2l-exercises',
        }
        resp = requests.get(url, headers=headers, timeout=3)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            with z.open('cmn.txt') as f:
                content = f.read()

        with open(filepath, 'wb') as f:
            f.write(content)

        return resp.text

    @staticmethod
    def _extract_lines_of_words(
        filepath: str
    ) -> tuple[list[list[str]], list[list[str]]]:
        """
        Extracts the lines of words from the TSV file that has three columns:
        - source text
        - target text
        - alignment

        Returns a tuple of two lists:
        - list #1: the source lines of words.
        - list #2: the target lines of words.
        """

        import csv

        source_corpus, target_corpus = [], []
        with open(filepath, 'r', encoding='UTF-8') as f:
            file = csv.reader(f, delimiter='\t', lineterminator='\n')
            for row in file:
                if len(row) != 3:
                    continue # The final newline

                source_corpus.append(
                    TextSequenceDataset.preprocess_source_text_sequence(row[0]))
                target_corpus.append(
                    TextSequenceDataset.preprocess_target_text_sequence(row[1]))

        return source_corpus, target_corpus

    @staticmethod
    def _extract_evaluation_samples(
        source_lines_of_words: list[list[str]],
        target_lines_of_words: list[list[str]],
        num: int = 1,
    ) -> list[tuple[str, str]]:
        assert len(source_lines_of_words) == len(target_lines_of_words)

        samples = []
        for _ in range(num):
            index = random.randint(0, len(source_lines_of_words))
            samples.append((
                ' '.join(source_lines_of_words[index]),
                ''.join(target_lines_of_words[index])
            ))

        return samples


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


class PositionalEncoding(nn.Module):
    def __init__(
        self,
        num_embedding_dims: int,
        dropout: float,
    ) -> None:
        """
        Parameters:
        - num_embedding_dims: the number of embedding dimensions.
        - dropout: the dropout rate.
        """
        super().__init__()

        self._P = self._new_positional_encoding(num_embedding_dims)
        self._dropout = nn.Dropout(dropout)

    def _new_positional_encoding(
        self,
        num_embedding_dims: int,
    ) -> torch.Tensor:
        """
        Computes the positional encoding for the given number of embedding dimensions.

        Parameters:
        - num_embedding_dims: the number of embedding dimensions.

        Returns the positional encoding tensor, with the shape of (num_embedding_dims,).
        """
        P = torch.repeat_interleave(
            torch.arange(0, num_embedding_dims/2, dtype=torch.float, device=device),
            2)
        return 1 / 10000 ** (2 * P / num_embedding_dims)

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - X: the input tensor with the shape of (batch_size, num_steps, num_embedding_dims)

        Returns the positional encoded tensor with the same shape as the input tensor.
        """

        assert_dimension('X', X, 3)
        assert X.shape[2] == self._P.shape[0], \
            (f"Mismatched shape at dimension 2 between X and self._P "
             f"({X.shape[2]} vs. {self._P.shape[0]})")

        num_steps = X.shape[1]
        P = self._P.unsqueeze(0).repeat(num_steps, 1)
        row_number = torch.arange(1, num_steps+1, device=device).reshape(-1, 1)
        P[:, 0::2] = torch.sin(P[:, 0::2] * row_number)
        P[:, 1::2] = torch.cos(P[:, 1::2] * row_number)
        assert_same_partial_shape('X', X, 'P', P, dim=(-2, -1))

        return self._dropout(X + P)


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


class PositionwiseFFN(nn.Module):
    def __init__(
        self,
        num_hidden_units: int,
        num_output_units: int,
    ) -> None:
        super().__init__()

        self._mlp = nn.Sequential(
            nn.LazyLinear(num_hidden_units, bias=False, device=device),
            nn.ReLU(),
            nn.LazyLinear(num_output_units, bias=False, device=device),
        )

    def forward(
        self,
        X: torch.Tensor,
    ) -> torch.Tensor:
        return self._mlp(X)


class AddNorm(nn.Module):
    def __init__(
        self,
        norm_shape: int | tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        self._norm = nn.LayerNorm(norm_shape, device=device) # type: ignore
        self._dropout = nn.Dropout(dropout)

    def forward(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> torch.Tensor:
        assert_same_shape('X', X, 'Y', Y)
        return self._norm(X + self._dropout(Y))


class EncoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_hidden_units: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self._mha = MultiHeadAttention(num_hidden_units, num_heads, dropout)
        self._addnorm1 = AddNorm(num_hidden_units, dropout)
        self._pffn = PositionwiseFFN(num_hidden_units, num_hidden_units)
        self._addnorm2 = AddNorm(num_hidden_units, dropout)

        self._num_hidden_units = num_hidden_units

        # For visualization
        self.attention_weights = torch.tensor([])

    def forward(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - inputs: a tuple of two tensors, in the following order:
          - the input tensor with the shape of (batch_size, num_steps, num_hidden_units).
          - the valid lengths of the input tensor with the shape of (batch_size,).

        Returns a tuple of two tensors:
          - the output tensor with the shape of (batch_size, num_steps, num_hidden_units).
          - the valid lengths of the output tensor with the shape of (batch_size,).
        """

        assert isinstance(inputs, tuple)
        assert len(inputs) == 2

        X, valid_lens = inputs

        assert_dimension('X', X, 3)
        assert_dimension('valid_lens', valid_lens, 1)
        assert_dimension_size('X', X, 2, self._num_hidden_units)
        # Ensure the same batch size
        assert_same_partial_shape('X', X, 'valid_lens', valid_lens, dim=0)

        outputs_shape = X.shape

        # Multi-head self-attention
        outputs, weights = self._mha(X, X, X, valid_lens)
        assert_shape('outputs', outputs, outputs_shape)
        outputs = self._addnorm1(X, outputs)
        assert_shape('outputs', outputs, outputs_shape)
        self.attention_weights = weights

        # Positionwise FFN
        X = outputs
        outputs = self._pffn(X)
        assert_shape('outputs', outputs, outputs_shape)
        outputs = self._addnorm2(X, outputs)
        assert_shape('outputs', outputs, outputs_shape)

        return (outputs, valid_lens)


class Encoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        num_hidden_units: int,
        dropout: float,
        vocab_size: int,
    ) -> None:
        super().__init__()

        self._embedding = nn.Embedding(vocab_size, num_hidden_units, device=device)
        self._positional_encoding = PositionalEncoding(num_hidden_units, dropout)
        self._blocks = nn.Sequential(*[EncoderBlock(num_heads, num_hidden_units, dropout)
                                      for _ in range(num_blocks)])

        self._num_hidden_units = num_hidden_units
        self._num_embedding_dims = num_hidden_units

        self.apply(_init_weights_fn)

    def forward(
        self,
        source: torch.Tensor,
        valid_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - source: the source tensor with the shape of (batch_size, num_steps).
        - valid_lens: the valid lengths of the source tensor with the shape of (batch_size,).

        Returns the output tensor with the shape of (batch_size, num_steps, num_hidden_units).
        """

        assert_dimension('source', source, 2)
        assert_dimension('valid_lens', valid_lens, 1)
        # Ensure the same batch size
        assert_same_partial_shape('source', source, 'valid_lens', valid_lens, dim=0)

        batch_size, num_steps = source.shape

        # Scale the embeddings before adding the positional encoding to balance the scale
        # of the values in the embeddings with the positional encoding.
        X = self._positional_encoding(
            self._embedding(source) * math.sqrt(self._num_embedding_dims))
        assert_shape('X', X, (batch_size, num_steps, self._num_embedding_dims))

        output, _ = self._blocks((X, valid_lens))
        assert_shape('output', output, (batch_size, num_steps, self._num_hidden_units))

        return output

    @property
    def attention_weights(self) -> torch.Tensor:
        weights = []
        for block in self._blocks:
            weights.append(block.attention_weights)

        return torch.stack(weights)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        num_heads: int,
        num_hidden_units: int,
        dropout: float,
    ) -> None:
        super().__init__()

        # The multi-head self-attentioin
        self._causal_mha = MultiHeadAttention(num_hidden_units, num_heads, dropout)
        self._addnorm1 = AddNorm(num_hidden_units, dropout)

        # The multi-head encoder-decoder attention
        self._cross_mha = MultiHeadAttention(num_hidden_units, num_heads, dropout)
        self._addnorm2 = AddNorm(num_hidden_units, dropout)

        # The positionwise forward feed network
        self._pffn = PositionwiseFFN(num_hidden_units, num_hidden_units)
        self._addnorm3 = AddNorm(num_hidden_units, dropout)

        # The Key-Value Cache. When set to None, cache is disabled.
        self._kv_cache: DecoderBlock.KVCache | None = None

        self._is_predicting: bool = False
        self._num_heads = num_heads
        self._num_hidden_units = num_hidden_units

        # For visualization
        self.causal_attention_weights = torch.tensor([])
        self.cross_attention_weights = torch.tensor([])

    def forward(
        self,
        intputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - inputs: a tuple of three tensors, in the following order:
          - the target tensor with the shape of (batch_size, num_steps, num_hidden_units).
          - the encoder outputs tensor with the shape of (batch_size, num_steps, num_hidden_units).
          - the valid lengths of the encoder outputs tensor with the shape of
            (batch_size,).

        Returns a tuple of three tensors:
          - the output tensor with the shape of (batch_size, num_steps, num_hidden_units).
          - the encoder outputs tensor with the shape of (batch_size, num_steps, num_hidden_units).
          - the valid lengths of the encoder outputs tensor with the shape of
            (batch_size,).
        """

        assert isinstance(intputs, tuple)
        assert len(intputs) == 3

        X, encoder_output, encoder_valid_lens = intputs

        assert_dimension('X', X, 3)
        assert_dimension('encoder_output', encoder_output, 3)
        assert_dimension('encoder_valid_lens', encoder_valid_lens, 1)
        # Ensure the same batch size and hidden size
        assert_same_partial_shape('X', X, 'encoder_output', encoder_output, dim=(0, 2))
        assert_dimension_size('X', X, 2, self._num_hidden_units)

        batch_size, num_steps, _ = outputs_shape = X.shape

        # Causal self-attention
        if self.is_predicting():
            assert self._kv_cache is not None
            assert_dimension_size('X', X, dim=(0, 1), size=(1, 1))

            # During prediction, no masking is needed as the query naturally can't attend to
            # future tokens that haven't been generated yet.
            decoder_valid_lens = None

            keys = values = self._kv_cache.add(X)
        else:
            assert self._kv_cache is None

            # During training, each row in X represents a complete sequences. So we mask all the
            # positions after the query position to ensure the query only attends to itself
            # and previous tokens.
            decoder_valid_lens = torch.arange(start=1, end=num_steps + 1, device=device)
            decoder_valid_lens = decoder_valid_lens.unsqueeze(0).expand(batch_size, -1)
            assert_shape('decoder_valid_lens', decoder_valid_lens, (batch_size, num_steps))

            keys = values = X

        outputs, weights = self._causal_mha(X, keys, values, decoder_valid_lens)
        assert_shape('outputs', outputs, outputs_shape)
        outputs = self._addnorm1(X, outputs)
        assert_shape('outputs', outputs, outputs_shape)
        self.causal_attention_weights = weights

        # Cross-attention
        X = outputs
        keys = values = encoder_output
        outputs, weights = self._cross_mha(X, keys, values, encoder_valid_lens)
        assert_shape('outputs', outputs, outputs_shape)
        outputs = self._addnorm2(X, outputs)
        assert_shape('outputs', outputs, outputs_shape)
        self.cross_attention_weights = weights

        # Positionwise FFN
        X = outputs
        outputs = self._pffn(X)
        assert_shape('outputs', outputs, outputs_shape)
        outputs = self._addnorm3(X, outputs)
        assert_shape('outputs', outputs, outputs_shape)

        return (outputs, encoder_output, encoder_valid_lens)

    def is_predicting(self) -> bool:
        return self._is_predicting

    def begin_predicting(self) -> None:
        self._is_predicting = True
        self._kv_cache = DecoderBlock.KVCache()

    def end_predicting(self) -> None:
        self._kv_cache = None
        self._is_predicting = False


    class KVCache:
        def __init__(self) -> None:
            self._data: torch.Tensor | None = None

        def add(
            self,
            X: torch.Tensor
        ) -> torch.Tensor:
            if self._data is None:
                self._data = X
            else:
                assert_shape('X', X, (1, 1, self._data.size(2)))
                self._data = torch.cat((self._data, X), dim=1) # along the num_steps dimension

            return self._data

        def length(self) -> int:
            return self._data.size(1) if self._data is not None else 0


class Decoder(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        num_hidden_units: int,
        dropout: float,
        vocab_size: int,
    ) -> None:
        super().__init__()

        self._embedding = nn.Embedding(vocab_size, num_hidden_units, device=device)
        self._positional_encoding = PositionalEncoding(num_hidden_units, dropout)
        self._blocks = nn.Sequential(*[DecoderBlock(num_heads, num_hidden_units, dropout)
                                      for _ in range(num_blocks)])
        self._output_layer = nn.LazyLinear(vocab_size, device=device)

        self._num_heads = num_heads
        self._num_hidden_units = num_hidden_units
        self._num_embedding_dims = num_hidden_units
        self._vocab_size = vocab_size

        self.apply(_init_weights_fn)

    def forward(
        self,
        X: torch.Tensor,
        encoder_outputs: torch.Tensor,
        encoder_valid_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - X: the target tensor with the shape of (batch_size, num_steps).
        - encoder_outputs: the encoder outputs tensor with the shape of
            (batch_size, num_steps, num_hidden_units).
        - encoder_valid_lens: the valid lengths of the encoder outputs tensor with the
            shape of (batch_size,).

        Returns the output tensor with the shape of (batch_size, num_steps, vocab_size).
        """

        assert_dimension('X', X, 2)
        assert_dimension('encoder_outputs', encoder_outputs, 3)
        assert_dimension_size('encoder_outputs', encoder_outputs, -1, self._num_hidden_units)
        assert_dimension('encoder_valid_lens', encoder_valid_lens, 1)
        # Ensure the same batch size
        assert_same_partial_shape('X', X, 'encoder_outputs', encoder_outputs, dim=0)
        assert_same_partial_shape(
            'encoder_outputs', encoder_outputs, 'encoder_valid_lens', encoder_valid_lens, dim=0)

        batch_size, num_steps = X.shape

        X = self._embedding(X)
        assert_shape('X', X, (batch_size, num_steps, self._num_embedding_dims))

        X = self._positional_encoding(X * math.sqrt(X.size(2)))
        assert_shape('X', X, (batch_size, num_steps, self._num_embedding_dims))

        outputs, _, _ = self._blocks((X, encoder_outputs, encoder_valid_lens))
        assert_shape('outputs', outputs, (batch_size, num_steps, self._num_hidden_units))

        outputs = self._output_layer(outputs)
        assert_shape('outputs', outputs, (batch_size, num_steps, self._vocab_size))

        return outputs

    def begin_predicting(
        self,
    ) -> None:
        for block in self._blocks:
            block.begin_predicting()

    def end_predicting(
        self,
    ) -> None:
        for block in reversed(self._blocks):
            block.end_predicting()

    @property
    def attention_weights(self) -> tuple[torch.Tensor, torch.Tensor]:
        causal_attention_weights, cross_attention_weights = [], []
        for block in self._blocks:
            causal_attention_weights.append(block.causal_attention_weights)
            cross_attention_weights.append(block.cross_attention_weights)

        return (torch.stack(causal_attention_weights),
                torch.stack(cross_attention_weights))


class TransformerTranslator(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        num_heads: int,
        num_hidden_units: int,
        dropout: float,
        grad_clip_threshold: float,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        super().__init__()

        self._encoder = Encoder(num_blocks, num_heads, num_hidden_units, dropout, source_vocab.size)
        self._decoder = Decoder(num_blocks, num_heads, num_hidden_units, dropout, target_vocab.size)

        self._num_hidden_units = num_hidden_units
        self._grad_clip_threshold = grad_clip_threshold
        self._source_vocab = source_vocab
        self._target_vocab = target_vocab

    def forward(
        self,
        source: torch.Tensor,
        source_valid_lens: torch.Tensor,
        target_X: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - source: the source tensor with the shape of (batch_size, num_steps).
        - source_valid_lens: the valid lengths of the source tensor with the shape of (batch_size,).
        - target_X: the target tensor with the shape of (batch_size, num_steps).

        Returns the output tensor with the shape of (batch_size, num_steps, vocab_size).
        """

        assert_dimension('source', source, 2)
        assert_dimension('source_valid_lens', source_valid_lens, 1)
        assert_dimension('target_X', target_X, 2)
        # Ensure the same batch size
        assert_same_partial_shape('source', source, 'source_valid_lens', source_valid_lens, dim=0)
        assert_same_partial_shape('source', source, 'target_X', target_X, dim=0)

        batch_size, num_steps = source.shape
        output = self._encoder(source, source_valid_lens)
        assert_shape('output', output, (batch_size, num_steps, self._num_hidden_units))

        batch_size, num_steps = target_X.shape
        output = self._decoder(target_X, output, source_valid_lens)
        assert_shape('output', output, (batch_size, num_steps, self._target_vocab.size))

        return output

    def predict(
        self,
        sentence: str,
        max_length: int = 10,
    ) -> tuple[str, tuple[torch.Tensor, ...]]:
        """
        Parameters:
        - sentence: the source sentence.
        - max_length: the maximum length of the target sequence.

        Returns a tuple of two tensors:
        - the predicted target sequence.
        - a tuple of three tensors of attention weights, in the following order:
          - the encoder's attention weights
          - the decoder's causal-attention weights
          - the decoder's cross-attention weights
        """

        self.eval()
        with torch.inference_mode():
            return self._predict(sentence, max_length)

    def _predict(
        self,
        sentence: str,
        max_length: int,
    ) -> tuple[str, tuple[torch.Tensor, ...]]:
        batch_size: Final = 1
        num_steps: Final = 1
        source_vocab, target_vocab = self._source_vocab, self._target_vocab

        words = TextSequenceDataset.preprocess_source_text_sequence(sentence)
        source, source_valid_lens = TextSequenceDataset.tokenize([words], source_vocab)
        assert_shape('source', source, (batch_size, int(source_valid_lens.item())))
        assert source_valid_lens.item() == len(words) + 1 # <eos>

        encoder_output = self._encoder(source, source_valid_lens)
        assert_shape('encoder_output', encoder_output,
                     (batch_size, int(source_valid_lens.item()), self._num_hidden_units))

        decoder_attention_weights = []
        target_X = torch.tensor([target_vocab.bos_token], device=device).view(batch_size, num_steps)
        prediction = None

        self._decoder.begin_predicting()
        for _ in range(max_length):
            output = self._decoder(target_X, encoder_output, source_valid_lens)
            assert_shape('output', output, (batch_size, num_steps, target_vocab.size))
            decoder_attention_weights.append(self._decoder.attention_weights)

            target_Y = output.argmax(2)
            assert_shape('target_Y', target_Y, (batch_size, num_steps))
            if target_Y[0][0].item() == target_vocab.eos_token:
                logger.debug("<eos> generated")
                break

            # Save the predicted token
            if prediction is None:
                prediction = target_Y
            else:
                # along the num_steps dimension
                prediction = torch.cat((prediction, target_Y), dim=1)

            target_X = target_Y
        self._decoder.end_predicting()

        assert prediction is not None
        lines_of_words = TextSequenceDataset.untokenize(prediction, target_vocab)
        assert len(lines_of_words) == batch_size

        # Ensure both causal and cross attention weights are 4D tensors.
        assert_dimension('decoder_attention_weights[0][0]', decoder_attention_weights[0][0], 4)
        assert_dimension('decoder_attention_weights[0][1]', decoder_attention_weights[0][1], 4)

        return (
            ''.join(lines_of_words[0]),
            (
                self._encoder.attention_weights,
                *self.prepare_decoder_attention_weights(decoder_attention_weights)
            )
        )

    def prepare_decoder_attention_weights(
        self,
        weights: list[tuple],
    ) -> tuple[torch.Tensor, torch.Tensor]:

        dim_num_queries: Final = 2
        dim_num_keys: Final = 3
        causal_weights = []
        cross_weights = []
        for w in weights:
            assert_dimension('w[0]', w[0], 4)
            assert_dimension_size('w[0]', w[0], dim=dim_num_queries, size=1)
            causal_weights.append(w[0])

            assert_dimension('w[1]', w[1], 4)
            assert_dimension_size('w[1]', w[1], dim=dim_num_queries, size=1)
            cross_weights.append(w[1])

        max_num_keys = causal_weights[-1].size(dim_num_keys)
        for i, w in enumerate(causal_weights[:-1]):
            num_paddings = max_num_keys - w.size(dim_num_keys)
            w = torch.nn.functional.pad(w, (0, num_paddings), 'constant', 0)
            causal_weights[i] = w

        return (torch.cat(causal_weights, dim=dim_num_queries),
                torch.cat(cross_weights, dim=dim_num_queries))

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


class LossMeasurer:
    def __init__(
        self,
        vocab: Vocabulary,
    ) -> None:
        self._vocab = vocab

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """
        Measures the loss using the cross-entropy loss function.

        Parameters:
        - prediction: the prediction tensor with the shape of (batch_size, num_steps, vocab_size).
        - target: the target tensor with the shape of (batch_size, num_steps).

        Returns the loss tensor.
        """

        assert_dimension('predictions', prediction, 3)
        batch_size, num_steps, vocab_size = prediction.shape
        assert vocab_size == self._vocab.size, \
            f"mismatched final dimension, got {vocab_size}, expected {self._vocab.size}"
        assert_shape('labels', target, (batch_size, num_steps))

        Y = target.view(-1)
        assert_shape('Y', Y, (batch_size * num_steps, ))
        Y_pred = prediction.view(-1, vocab_size)
        assert_shape('Y_pred', Y_pred, (batch_size * num_steps, vocab_size))

        mask = Y != self._vocab.pad_token

        # **Filter out** the padded positions in the labels and predictions
        masked_Y = Y[mask]
        masked_Y_pred = Y_pred[mask]
        # CAUTION:
        #   Avoid reshaping or repeating the mask into the shape of either
        #   (batch_size * num_steps, ) or ((batch_size * num_steps, vocab_size),
        #   and applying it through element-wise multiplication.
        #   Doing so will distort the prediction and target (labels) tensors.
        #   The masked (zeroed) rows no longer carry meaningful information
        #   for classification and will mislead the loss function.
        #
        #   The reason I know this involves a tale of woe, myriad cups of tea,
        #   and a long night of debugging :-)

        return nn.functional.cross_entropy(masked_Y_pred, masked_Y)


class Trainer:
    def __init__(
        self,
        model: TransformerTranslator,
        dataset: TextSequenceDataset,
        loss_measurer: LossMeasurer,
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

        dataloader = self._dataset.get_data_loader(train=True, shuffle=False)
        for source, source_valid_lens, target_X, _, y in dataloader:
            source = source.to(device)
            source_valid_lens = source_valid_lens.to(device)
            target_X = target_X.to(device)
            y_indices = y.to(device)

            y_pred_logits = self._model(source, source_valid_lens, target_X)

            loss = self._loss_measurer(y_pred_logits, y_indices)

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
        model: TransformerTranslator,
        dataset: TextSequenceDataset,
        loss_measurer: LossMeasurer,
    ) -> None:
        self._model = model
        self._dataset = dataset
        self._loss_measurer = loss_measurer

    def __call__(self) -> tuple[float, float]:
        """
        Validates the model.

        Returns a tuple of two float values:
        - the mean loss
        - the perplexity
        """
        num_batches = 0
        total_loss = 0.0

        self._model.eval()

        dataloader = self._dataset.get_data_loader(train=False, shuffle=True)
        with torch.inference_mode():
            for source, source_valid_lens, target_X, _, y in dataloader:
                source = source.to(device)
                source_valid_lens = source_valid_lens.to(device)
                target_X = target_X.to(device)
                y_indices = y.to(device) # shape is (batch_size, num_steps)

                y_pred_logits = self._model(source, source_valid_lens, target_X)

                total_loss += self._loss_measurer(y_pred_logits, y_indices).item()
                num_batches += 1

        mean_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(mean_loss, device=device)).item()
        return (mean_loss, perplexity)

    def index(self, y: torch.Tensor) -> torch.Tensor:
        return y.argmax(dim=-1)


class BLEU:
    def __init__(
        self,
        prediction: str,
        target: str,
        k: int = 2,
    ) -> None:
        self._prediction = prediction
        self._target = target
        self._k = k

    @property
    def score(self) -> float:
        prediction_words = TextSequenceDataset.preprocess_target_text_sequence(self._prediction)
        prediction_len = len(prediction_words)
        if prediction_len == 0:
            # Sometimes the model predicts an empty sequence, especially at the beginning
            # of training. In such cases, the BLEU score is 0.0.
            return 0.0

        target_words = TextSequenceDataset.preprocess_target_text_sequence(self._target)
        target_len = len(target_words)

        precisions = 1.0
        for n in range(1, min(self._k, prediction_len, target_len) + 1):
            p = self._precision(prediction_words, target_words, n)
            precisions *= p**(1/(2**n))

        penalty = self._penalty(prediction_len, target_len)
        return penalty * precisions

    @staticmethod
    def _precision(
        prediction_words: list[str],
        target_words: list[str],
        n: int,
    ) -> float:
        prediction_sequences = BLEU._ngram(prediction_words, n)
        target_sequences = BLEU._ngram(target_words, n)

        num_matched, num_total = 0, len(prediction_sequences)
        for seq in prediction_sequences:
            if seq in target_sequences:
                num_matched += 1
                target_sequences.remove(seq)

        return float(num_matched) / num_total

    @staticmethod
    def _penalty(
        prediction_len: int,
        target_len: int
    ) -> float:
        return math.exp(min(0, 1 - float(target_len)/prediction_len))

    @staticmethod
    def _ngram(
        words: list[str],
        n: int,
    ) -> list[tuple[str, ...]]:
        assert n < len(words)

        sequences = []
        for i in range(len(words)):
            if i + n > len(words):
                break

            sequences.append(tuple(words[i:i+n]))

        return sequences


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
            f"Invalid type for perplexity (expected 'float', got {type(perplexity)}"

        self._epochs.append(epoch)
        self._train_losses.append(train_loss)
        self._evaluate_losses.append(validate_loss)
        self._perplexities.append(perplexity)

    def plot(
        self,
        title: str,
        filename: str = "metrics.jpg",
    ) -> None:
        """
        Plots the training and validation loss, and the perplexity.

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


def preview(
    dataset: TextSequenceDataset,
    num: int,
) -> None:
    if num <= 0:
        return

    total = 0
    for source, source_valid_lens, target_X, _, target_Y in dataset.get_data_loader():
        assert_dimension('source', source, 2)
        assert_dimension('source_valid_lens', source_valid_lens, 1)
        assert_dimension('target_X', target_X, 2)
        assert_dimension('target_Y', target_Y, 2)

        for i, (s, s_l, t_x, t_y) in enumerate(zip(source, source_valid_lens, target_X, target_Y)):
            s = dataset.source_vocab.untokenize(s.tolist())
            t_x = dataset.target_vocab.untokenize(t_x.tolist())
            t_y = dataset.target_vocab.untokenize(t_y.tolist())

            logger.info("preview #{}:", i)
            logger.info("    source   = {}", s)
            logger.info("    source valid len = {}", s_l)
            logger.info("    target_x = {}", t_x)
            logger.info("    target_y = {}", t_y)

            total += 1
            if total >= num:
                return


def evaluate(
    dataset: TextSequenceDataset,
    model: TransformerTranslator,
) -> tuple[list[float], tuple[torch.Tensor, ...]]:
    scores = []
    attention_weight = (torch.empty(0), )
    for i, (source, target) in enumerate(dataset.evaluation_samples):
        prediction, attention_weight = model.predict(source)
        score = BLEU(prediction, target).score
        scores.append(score)

        logger.info("prediction: #{}", i)
        logger.info("    source = '{}'", source)
        logger.info("    target = '{}'", target)
        logger.info("    prediction = '{}'", prediction)
        logger.info("    bleu = {:.2f}", score)

    return scores, attention_weight


def main(
    num_preview: int = 1
) -> None:
    max_epochs = 30
    learning_rate = 0.001
    batch_size = 128
    num_blocks = 2
    num_heads = 4
    num_hidden_units = 256
    weight_decay = 1e-5
    dropout = 0.2
    grad_clip_threshold = 1.0
    num_evaluation_samples = 4
    start_sec = time.time()

    # Print hyperparameters
    logger.info("max_epochs = {}, learning_rate = {:.3f}, batch_size = {}, device = {}",
                max_epochs, learning_rate, batch_size, device)
    logger.info("weight_decay = {:.1e}, dropout = {:.2f}, grad_clip_threshold = {:.2f}",
             weight_decay, dropout, grad_clip_threshold)
    logger.info("num_blocks = {}, num_heads = {}, num_hidden_units = {}",
                num_blocks, num_heads, num_hidden_units)

    # Initialize dataset
    dataset = TextSequenceDataset(
        "./chinese_english.txt", batch_size, num_evaluation_sample=num_evaluation_samples)
    preview(dataset, num_preview)

    logger.debug("source vocab = {}", dataset.source_vocab)
    logger.debug("target vocab = {}", dataset.target_vocab)

    # Initialize model, loss_measurer, and optimizer
    model = TransformerTranslator(num_blocks, num_heads, num_hidden_units, dropout,
        grad_clip_threshold, dataset.source_vocab, dataset.target_vocab)
    loss_measurer = LossMeasurer(dataset.target_vocab)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    # Initialize trainer and validator
    trainer = Trainer(model, dataset, loss_measurer, optimizer)
    validator = Validator(model, dataset, loss_measurer)

    # Initialize plotter and visualizer
    plotter = MetricsPlotter()
    visualizer = AttentionWeigthsVisualizer()

    # Train, validate and evaluate
    for epoch in range(max_epochs):
        train_loss = trainer.fit()
        validate_loss, perplexity = validator()
        scores, attention_weights = evaluate(dataset, model)

        plotter.add(epoch, train_loss, validate_loss, perplexity)
        visualizer.add(epoch, *attention_weights)

        logger.info("epoch #{}, train_loss = {:.3f}, validate_loss = {:.3f}, perplexity = {:.2f}, "
                    "bleu_scores = {}",
                    epoch, train_loss, validate_loss, perplexity,
                    [f'{s:.2f}' for s in scores])

    logger.info("device = {}, elapsed time: {:.1f} seconds", device, time.time() - start_sec)

    # Plot metrics
    plotter.plot(title=f"Machine Translation ({TransformerTranslator.__name__})")

    # Visualize attention weights
    visualizer(start=max_epochs-1)

    logger.info("done!")


if __name__ == "__main__":
    # Hi everyone.
    #
    # I've been struggling with this model because it tends to generate tokens until it reaches
    # the maximum length, rather than producing an end-of-sentence token.
    # I've already spent over ten hours trying to resolve it but haven't succeeded yet. I'd be
    # deeply appreciate any help in figuring this out.
    #
    # Thank you so much in advance.

    logger.remove()
    logger.add(sink=sys.stderr, level="INFO")

    main(num_preview=3)

    # prediction: #0
    #     source = 'the dog is sitting by the bowl .'
    #     target = '狗坐在碗旁边。'
    #     prediction = '狗坐在碗旁边。'
    #     bleu = 1.00
    # prediction: #1
    #     source = 'tom's funeral will be this weekend .'
    #     target = '湯姆的葬禮訂在這週末。'
    #     prediction = '湯姆的葬禮訂禮訂禮訂'
    #     bleu = 0.61
    # prediction: #2
    #     source = 'this knife is very sharp .'
    #     target = '这把刀很锋利。'
    #     prediction = '这把刀子很锋利用刀子'
    #     bleu = 0.63
    # prediction: #3
    #     source = 'please tell me tom is ok .'
    #     target = '请告诉我汤姆很好。'
    #     prediction = '請告訴我告訴湯姆跟我'
    #     bleu = 0.00
    # epoch #29, train_loss = 0.481, validate_loss = 4.152, perplexity = 63.57, bleu_scores = ['1.00', '0.61', '0.63', '0.00']
    # device = cuda, elapsed time: 99.9 seconds
    # done!
