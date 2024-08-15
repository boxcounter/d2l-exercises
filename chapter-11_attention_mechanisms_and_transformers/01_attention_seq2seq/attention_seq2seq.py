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
from typing import Literal, Final
import reprlib

import torch
import torch.utils
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
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


def assert_tensor(
    tensor_name: str,
    tensor,
) -> None:
    assert tensor is not None, f"{tensor_name} is None"
    assert isinstance(tensor, torch.Tensor), \
        f"Invalid type for {tensor_name} (expected 'torch.Tensor', got {type(tensor)})"


def assert_dimensions(
    tensor_name: str,
    tensor,
    expected_dimensions: int,
) -> None:
    assert_tensor(tensor_name, tensor)
    assert len(tensor.shape) == expected_dimensions, \
        (f"Invalid dimensions for {tensor_name} (expected {expected_dimensions}, "
         f"got {len(tensor.shape)})")


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
    dims: tuple[int, ...] | int,
) -> None:
    """
    Asserts that the two tensors have the same shape at the specified dimensions.
    """

    assert_tensor(tensor1_name, tensor1)
    assert_tensor(tensor2_name, tensor2)

    if isinstance(dims, int):
        dims = (dims, )

    for dim in dims:
        assert tensor1.shape[dim] == tensor2.shape[dim], \
            (f"Mismatched shape at dimension {dim} between {tensor1_name} and {tensor2_name} "
             f"({tensor1.shape[dim]} vs. {tensor2.shape[dim]})")


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
        if min_freq == 0:
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
            logger.debug("downloading content from {}", self.URL)
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
                torch.tensor(lines_of_valid_len, device = device))

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

        assert_dimensions('tokens', tokens, 2)
        tokens = tokens.to(dtype=torch.int)

        lines_of_words = []
        for t in tokens: # iterate for each batch
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
            'source', source, 'target', target, dims=0)
        assert_same_partial_shape(
            'source', source, 'source_valid_len', source_valid_len, dims=0)
        assert_same_partial_shape(
            'source_valid_len', source_valid_len, 'target_valid_len', target_valid_len, dims=0)

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


class Encoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        num_directions: Literal[1, 2],
        num_hidden_units: int,
        num_embedding_dims: int,
        dropout: float,
        vocab_size: int,
    ) -> None:
        """
        Parameters:
        - num_layers: the number of layers.
        - num_directions: the number of directions (1 or 2).
        - num_hidden_units: the number of hidden units.
        - num_embedding_dims: the number of embedding dimensions.
        - dropout: the dropout rate.
        - vocab_size: the size of the vocabulary.
        """

        super().__init__()

        self._embedding = nn.Embedding(
            vocab_size, num_embedding_dims, device=device)
        self._hidden_layer = nn.GRU(
            num_embedding_dims, num_hidden_units, num_layers, batch_first=True,
            dropout=dropout, bidirectional=(num_directions==2), device=device)

        self._num_layers = num_layers
        self._num_directions = num_directions
        self._num_hidden_units = num_hidden_units
        self._num_embedding_dims = num_embedding_dims

        self.apply(_init_weights_fn)

    def forward(
        self,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - source: the input (source) tensor with the shape of (batch_size, num_steps)

        Returns a tuple of two tensors:
        - tensor #1: the output with the shape of
            (batch_size, num_steps, num_directions * num_hidden_units)
        - tensor #2: the hidden state of the current time step, with the shape of
            (num_directions * num_layers, batch_size, num_hidden_units)
        """

        assert_dimensions('source', source, 2)
        batch_size, num_steps = source.shape
        num_layers, hidden_size, embedding_size, num_directions = (
            self._num_layers, self._num_hidden_units,
            self._num_embedding_dims, self._num_directions)

        X = self._embedding(source)
        assert_shape('X', X, (batch_size, num_steps, embedding_size))

        output, H = self._hidden_layer(X)
        assert_shape('output', output, (batch_size, num_steps, num_directions * hidden_size))
        assert_shape('H', H, (num_directions * num_layers, batch_size, hidden_size))
        return output, H


def masked_softmax(
    attention_scores: torch.Tensor,
    valid_lens: torch.Tensor,
) -> torch.Tensor:
    """
    Computes the softmax of the attention scores after masking the padding elements.

    Parameters:
    - attention_scores: the 3D tensor of attention scores with the shape of
        (batch_size, num_queries, num_keys)
    - valid_lens: the 1D tensor of valid lengths for the keys, with the shape of
        (batch_size)

    Returns the softmax of the attention scores.
    """
    assert_dimensions('attention_scores', attention_scores, 3)
    assert_same_partial_shape(
        'attention_scores', attention_scores, 'valid_lens', valid_lens, dims=0)
    assert_dimensions('valid_lens', valid_lens, 1)

    batch_size, num_queries, num_keys = attention_scores.shape
    mask = torch.arange(0, num_keys, device=device).repeat(batch_size, num_queries, 1)
    assert_shape('mask', mask, (batch_size, num_queries, num_keys))

    # Reshape valid_lens from (batch_size) to (batch_size, 1, 1) and repeat it for each query
    # and key to match the shape of mask
    valid_lens = valid_lens.reshape(-1, 1, 1).repeat(1, num_queries, num_keys)
    assert_same_shape('mask', mask, 'valid_len', valid_lens)
    mask = mask >= valid_lens # mask is True for the padding elements

    # Mask off the padding elements
    attention_scores[mask] = -1e6

    return nn.functional.softmax(attention_scores, dim=-1)


class AdditiveAttention(nn.Module):
    def __init__(
        self,
        num_hidden_units: int,
        dropout: float,
    ) -> None:
        """
        Parameters:
        - num_hidden_units: the number of hidden units.
        - dropout: the dropout rate.
        """
        super().__init__()

        self._W_q = nn.LazyLinear(num_hidden_units, bias=False, device=device)
        self._W_k = nn.LazyLinear(num_hidden_units, bias=False, device=device)
        self._w_v = nn.LazyLinear(1, bias=False, device=device)
        self._dropout = nn.Dropout(dropout)
        self._weights = torch.zeros(0, device=device)

    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        valid_lens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters:
        - queries: the queries with the shape of (batch_size, num_queries, num_hidden_units)
        - keys: the keys with the shape of (batch_size, num_keys, num_hidden_units)
        - values: the values with the shape of (batch_size, num_keys, num_features_of_values)
        - valid_lens: the valid lengths of the keys with the shape of (batch_size)

        Returns the attention with the shape of (batch_size, num_queries, num_features_of_values).
        """

        assert_dimensions('queries', queries, 3)
        assert_same_partial_shape('queries', queries, 'keys', keys, dims=0)
        assert_same_partial_shape('keys', keys, 'values', values, dims=(0, 1))
        assert_same_partial_shape('keys', keys, 'valid_lens', valid_lens, dims=0)

        queries = self._W_q(queries)
        keys = self._W_k(keys)

        batch_size, num_queries, _ = queries.shape
        _, num_keys, _ = keys.shape
        _, _, num_features_of_values = values.shape

        # Reshape queries and keys to enable broadcasting for attention score computation
        #
        # queries:
        # - from: (batch_size, num_queries, num_hidden_units)
        # - to:   (batch_size, num_queries, 1, num_hidden_units)
        #   The added dimension (1) aligns with the num_keys dimension in keys.
        #
        # keys:
        # - from: (batch_size, num_keys, num_hidden_units)
        # - to:   (batch_size, 1, num_keys, num_hidden_units)
        #   The added dimension (1) aligns with the num_queries dimension in queries.
        queries = queries.unsqueeze(2)
        keys = keys.unsqueeze(1)

        attention_score = self._w_v(torch.tanh(queries + keys))
        assert_shape('attention_score', attention_score, (batch_size, num_queries, num_keys, 1))

        attention_weights = masked_softmax(attention_score.squeeze(-1), valid_lens)
        assert_shape('attention_weights', attention_weights, (batch_size, num_queries, num_keys))
        self._weights = attention_weights

        # Perform batch matrix multiplication (BMM) between:
        # - attention_weights with the shape of (batch_size, num_queries, num_keys)
        # - values with the shape of (batch_size, num_values, num_features_of_values)
        #
        # The result will have the shape (batch_size, num_queries, num_features_of_values) , given
        # num_values equals to num_keys.
        attention = torch.bmm(self._dropout(attention_weights), values)
        assert_shape('attention', attention, (batch_size, num_queries, num_features_of_values))

        return attention

    @property
    def weights(self) -> torch.Tensor:
        return self._weights


class AttentionDecoder(nn.Module):
    def __init__(
        self,
        num_layers: int,
        encoder_num_directions: Literal[1, 2],
        encoder_num_hidden_units: int,
        decoder_num_hidden_units: int,
        num_embedding_dims: int,
        dropout: float,
        vocab_size: int,
    ) -> None:
        """
        Parameters:
        - num_layers: the number of layers.
        - encoder_num_directions: the number of directions in the encoder (1 or 2).
        - encoder_num_hidden_units: the number of hidden units in the encoder.
        - decoder_num_hidden_units: the number of hidden units in the decoder.
        - num_embedding_dims: the size of each embedding vector.
        - dropout: the dropout rate.
        - vocab_size: the size of the vocabulary.
        """

        super().__init__()

        self._attention = AdditiveAttention(decoder_num_hidden_units, dropout)
        self._embedding = nn.Embedding(
            vocab_size, num_embedding_dims, device=device)
        self._hidden_layer = nn.GRU(
            num_embedding_dims + encoder_num_directions * encoder_num_hidden_units,
            decoder_num_hidden_units,
            num_layers, batch_first=True, dropout=dropout, device=device)
        self._output_layer = nn.LazyLinear(vocab_size, device=device)

        self.apply(_init_weights_fn)

        self._num_layers = num_layers
        self._encoder_num_directions = encoder_num_directions
        self._encoder_num_hidden_units = encoder_num_hidden_units
        self._decoder_num_hidden_units = decoder_num_hidden_units
        self._num_embedding_dims = num_embedding_dims
        self._vocab_size = vocab_size

    def forward(
        self,
        target_X: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_H: torch.Tensor,
        valid_lens: torch.Tensor,
        H_prev: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters:
        - target_X: the input (target) tensor with the shape of (batch_size, num_steps)
        - encoder_output: the output of the encoder with the shape of
            (batch_size, num_steps_of_encoder, num_encoder_directions * num_hidden_units_of_encoder)
        - encoder_H: the hidden state at the final time step of the encoder, with the shape of
            (num_encoder_directions * num_layers, batch_size, num_hidden_units)
        - valid_lens: the valid lengths of the encoder output with the shape of (batch_size)
        - H_prev: the hidden state of the previous time step, with the shape of
            (num_layers, batch_size, num_hidden_units)

        Returns a tuple of two tensors:
        - tensor #1: the output with the shape of (batch_size, num_steps, vocab_size)
        - tensor #2: the hidden state of the current time step, with the shape of
            (num_layers, batch_size, num_hidden_units)
        """

        assert_dimensions('target_X', target_X, 2)
        batch_size, num_steps = target_X.shape
        num_layers, encoder_directions, encoder_hidden_size = (
            self._num_layers, self._encoder_num_directions, self._encoder_num_hidden_units)
        decoder_hidden_size, embedding_size, vocab_size = (
            self._decoder_num_hidden_units, self._num_embedding_dims, self._vocab_size)

        assert_dimensions('encoder_output', encoder_output, 3)
        assert_shape(
            'encoder_output', encoder_output,
            (batch_size, encoder_output.shape[1], encoder_directions * encoder_hidden_size))
        keys = values = encoder_output

        if H_prev is None:
            # Directly use the hidden state at the final time step of the encoder to
            # initialize the hidden state of the decoder.
            assert_shape('encoder_H', encoder_H,
                         (num_layers * encoder_directions, batch_size, encoder_hidden_size))

            if encoder_directions == 2:
                encoder_H = encoder_H.view(
                    num_layers, encoder_directions, batch_size, encoder_hidden_size)
                # Use the hidden states of the final layer from both directions
                encoder_H = torch.cat((encoder_H[:, 0], encoder_H[:, 1]), dim=-1)
                assert_shape('encoder_H', encoder_H,
                             (num_layers, batch_size, decoder_hidden_size))

            H = encoder_H
        else:
            assert_shape('H_prev', H_prev, (num_layers, batch_size, decoder_hidden_size))
            H = H_prev

        X = self._embedding(target_X)
        assert_shape('X', X, (batch_size, num_steps, embedding_size))

        SINGLE_STEP = SINGLE_QUERY = 1
        outputs = []

        for x in X.permute(1, 0, 2): # (num_steps, batch_size, embedding_size)
            assert_shape('x', x, (batch_size, embedding_size))
            x = x.unsqueeze(1)
            assert_shape('x', x, (batch_size, SINGLE_STEP, embedding_size))

            query = H[-1] # Only the final layer's hidden state is needed
            assert_shape('query', query, (batch_size, decoder_hidden_size))
            query = query.unsqueeze(1)
            assert_shape('query', query, (batch_size, SINGLE_QUERY, decoder_hidden_size))

            context = self._attention(query, keys, values, valid_lens)
            assert_shape('context', context, (batch_size, SINGLE_QUERY, decoder_hidden_size))

            output, H = self._hidden_layer(torch.concat((x, context), dim=-1), H)
            assert_shape('output', output, (batch_size, SINGLE_STEP, decoder_hidden_size))
            assert_shape('H', H, (num_layers, batch_size, decoder_hidden_size))

            outputs.append(output)

        outputs = torch.cat(outputs, dim=1) # along the num_steps dimension
        assert_shape('outputs', outputs, (batch_size, num_steps, decoder_hidden_size))

        prediction = self._output_layer(outputs)
        assert_shape('prediction', prediction, (batch_size, num_steps, vocab_size))

        return prediction, H

    @property
    def attention_weights(self) -> torch.Tensor:
        return self._attention.weights


class AttentionSeq2SeqTranslator(nn.Module):
    def __init__(
        self,
        num_layers: int,
        encoder_num_directions: Literal[1, 2],
        num_hidden_units: int,
        num_embedding_features: int,
        dropout: float,
        grad_clip_threshold: float,
        source_vocab: Vocabulary,
        target_vocab: Vocabulary,
    ) -> None:
        """
        Parameters:
        - num_layers: the number of layers.
        - encoder_num_directions: the number of directions in the encoder (1 or 2).
        - num_hidden_units: the number of hidden units.
        - num_embedding_features: the number of embedding features.
        - dropout: the dropout rate.
        - grad_clip_threshold: the gradient clipping threshold.
        - source_vocab: the source vocabulary.
        - target_vocab: the target vocabulary.
        """
        super().__init__()

        encoder_num_hidden_units = num_hidden_units
        decoder_num_hidden_units = num_hidden_units * encoder_num_directions

        self._encoder = Encoder(
            num_layers, encoder_num_directions, encoder_num_hidden_units,
            num_embedding_features, dropout, source_vocab.size)
        self._decoder = AttentionDecoder(
            num_layers, encoder_num_directions, encoder_num_hidden_units, decoder_num_hidden_units,
            num_embedding_features, dropout, target_vocab.size)

        self._num_layers = num_layers
        self._encoder_num_directions = encoder_num_directions
        self._encoder_num_hidden_units = encoder_num_hidden_units
        self._decoder_num_hidden_units = decoder_num_hidden_units
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
        - source: the input (source) tensor with the shape of (batch_size, num_steps)
        - source_valid_lens: the valid lengths of the source with the shape of (batch_size)
        - target_X: the input (target) tensor with the shape of (batch_size, num_steps)

        Returns the output tensor with the shape of (batch_size, num_steps, target_vocab_size).
        """

        assert_dimensions('source', source, 2)
        assert_dimensions('source_valid_lens', source_valid_lens, 1)
        assert_dimensions('target_X', target_X, 2)
        # Ensure the same batch_size
        assert_same_partial_shape('target_X', target_X, 'source', source, dims=0)
        source_batch_size, source_num_steps = source.shape

        num_layers, encoder_directions, encoder_hidden_size, decoder_hidden_size = (
            self._num_layers, self._encoder_num_directions, self._encoder_num_hidden_units,
            self._decoder_num_hidden_units)

        batch_size, _ = source.shape
        encoder_output, encoder_H = self._encoder(source)
        assert_shape(
            'encoder_output', encoder_output,
            (source_batch_size, source_num_steps, encoder_directions*encoder_hidden_size))
        assert_shape(
            'encoder_H', encoder_H,
            (encoder_directions*num_layers, batch_size, encoder_hidden_size))

        _, num_steps = target_X.shape
        output, H = self._decoder(target_X, encoder_output, encoder_H, source_valid_lens)
        assert_shape('output', output, (batch_size, num_steps, self._target_vocab.size))
        assert_shape('H', H, (num_layers, batch_size, decoder_hidden_size))

        # Although the assert_shape function confirms them as torch.Tensors, I still need
        # to use assert here to prevent Pyright errors.
        assert isinstance(output, torch.Tensor)

        return output

    def predict(
        self,
        sentence: str,
        max_length: int = 100,
    ) -> str:
        """
        Predicts the target sequence for the source sequence.

        Parameters:
        - sentence: the source sentence.
        - max_length: the maximum length of the target sequence.

        Returns the predicted target sequence.
        """
        with torch.inference_mode():
            return self._predict(sentence, max_length)

    def _predict(
        self,
        sentence: str,
        max_length: int,
    ) -> str:
        batch_size, num_steps = 1, 1
        source_vocab, target_vocab = self._source_vocab, self._target_vocab
        num_layers, encoder_directions, encoder_hidden_size = (
            self._num_layers, self._encoder_num_directions, self._encoder_num_hidden_units)

        words = TextSequenceDataset.preprocess_source_text_sequence(sentence)
        source, source_valid_lens = TextSequenceDataset.tokenize([words], source_vocab)
        assert_shape('source', source, (batch_size, int(source_valid_lens.item())))
        assert source_valid_lens.item() == len(words) + 1 # <eos>

        encoder_output, encoder_H = self._encoder(source)
        assert_shape(
            'encoder_output', encoder_output,
            (batch_size, int(source_valid_lens.item()), encoder_directions * encoder_hidden_size))
        assert_shape(
            'encoder_H', encoder_H,
            (encoder_directions * num_layers, batch_size, encoder_hidden_size))

        target_X = torch.tensor([target_vocab.bos_token], device=device).view(1, -1)
        assert_dimensions('target_X', target_X, 2) # (batch_size, num_step)

        H = None

        for i in range(max_length):
            inputs = target_X[:, -1:]
            assert_shape('inputs', inputs, (batch_size, num_steps))

            output, H = self._decoder(inputs, encoder_output, encoder_H, source_valid_lens, H)
            assert_shape('output', output, (batch_size, num_steps, target_vocab.size))

            token_pred = output.argmax(dim=-1)
            assert_shape('token_pred', token_pred, (batch_size, num_steps))
            if token_pred[0][0] == target_vocab.eos_token:
                break

            target_X = torch.cat((target_X, token_pred), dim=-1) # along the num_steps dimension
            assert_shape('target_X', target_X, (batch_size, num_steps+i+1))

        tokens = target_X[:, 1:] # skip <bos>
        lines_of_words = TextSequenceDataset.untokenize(tokens, target_vocab)
        assert len(lines_of_words) == batch_size

        return ''.join(lines_of_words[0])

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

        logger.debug("gradients clipped, norm = {:.2f}, clip_ratio = {:.2f}",
                     norm.item(), clip_ratio.item())

    @property
    def attention_weights(self) -> torch.Tensor:
        return self._decoder.attention_weights


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

        assert_dimensions('predictions', prediction, 3)
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
        model: AttentionSeq2SeqTranslator,
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
        model: AttentionSeq2SeqTranslator,
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
        - value #1: the mean loss
        - value #2: the perplexity
        """
        num_batches = 0
        total_loss = 0.0

        self._model.eval()

        dataloader = self._dataset.get_data_loader(train=False, shuffle=True)
        with torch.inference_mode():
            for source, source_valid_lens, target_X, _, y in dataloader:
                source = source.to(device)
                target_X = target_X.to(device)
                y_indices = y.to(device) # shape is (batch_size, num_steps)

                y_pred_logits = self._model(source, source_valid_lens, target_X)

                total_loss += self._loss_measurer(y_pred_logits, y_indices).item()
                num_batches += 1

        mean_loss = total_loss / num_batches
        perplexity = torch.exp(torch.tensor(mean_loss)).item()
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


def plot_attention_weights_heatmap(
    attention_weights: torch.Tensor,
    sample_index: int = 0,
    title: str = "Attention Weights",
    filename: str = "attention_weights.jpg",
    cmap: str ='viridis',
) -> None:
    """
    Plots the heatmap for the attention weights of a single sample.

    Parameters:
    - attention_weights: tensor of shape (batch_size, num_queries, num_keys)
    - sample_index: index of the sample in the batch to visualize
    - title: title of the heatmap
    - cmap: colormap used for the heatmap
    """

    attention_weights = attention_weights.detach().cpu().numpy()
    sample_weights = attention_weights[sample_index, :, :]

    plt.figure(figsize=(10, 8))
    plt.imshow(sample_weights, aspect='auto', cmap=cmap)  # auto aspect for better scaling
    plt.colorbar()
    plt.title(title)
    plt.ylabel('Queries')
    plt.xlabel('Keys')
    plt.savefig(filename)
    plt.show()


def preview(
    dataset: TextSequenceDataset,
    num: int,
) -> None:
    if num <= 0:
        return

    total = 0
    for source, source_valid_lens, target_X, _, target_Y in dataset.get_data_loader():
        assert_dimensions('source', source, 2)
        assert_dimensions('source_valid_lens', source_valid_lens, 1)
        assert_dimensions('target_X', target_X, 2)
        assert_dimensions('target_Y', target_Y, 2)

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
    model: AttentionSeq2SeqTranslator,
    print_log: bool = False,
) -> list[float]:
    scores = []
    for i, (source, target) in enumerate(dataset.evaluation_samples):
        prediction = model.predict(source)
        score = BLEU(prediction, target).score
        scores.append(score)

        if print_log:
            logger.info("prediction: #{}", i)
            logger.info("    source = '{}'", source)
            logger.info("    target = '{}'", target)
            logger.info("    prediction = '{}'", prediction)
            logger.info("    bleu = {:.2f}", score)

    return scores


def main(
    num_preview: int = 1
) -> None:
    max_epochs = 30
    learning_rate = 0.005
    batch_size = 128
    num_layers = 2
    encoder_num_directions = 2
    num_hidden_units = 128
    num_embedding_dims = 256
    weight_decay = 1e-5
    dropout = 0.2
    grad_clip_threshold = 1.0
    num_evaluation_samples = 4
    start_sec = time.time()

    # Print hyperparameters
    logger.info(
        "max_epochs = {}, learning_rate = {:.3f}, batch_size = {}, device = {}",
        max_epochs, learning_rate, batch_size, device)
    logger.info(
        "weight_decay = {:.1e}, dropout = {:.2f}, grad_clip_threshold = {:.2f}",
        weight_decay, dropout, grad_clip_threshold)
    logger.info(
        "num_layers = {}, encoder_num_directions = {}, "
        "num_hidden_units = {}, num_embedding_dims = {}",
        num_layers, encoder_num_directions, num_hidden_units, num_embedding_dims)

    # Initialize dataset
    dataset = TextSequenceDataset(
        "./chinese_english.txt", batch_size, num_evaluation_sample=num_evaluation_samples)
    preview(dataset, num_preview)

    logger.debug("source vocab = {}", dataset.source_vocab)
    logger.debug("target vocab = {}", dataset.target_vocab)

    # Initialize model, loss, and optimizer
    model = AttentionSeq2SeqTranslator(
        num_layers, encoder_num_directions, num_hidden_units, num_embedding_dims, dropout,
        grad_clip_threshold, dataset.source_vocab, dataset.target_vocab)
    loss_measurer = LossMeasurer(dataset.target_vocab)
    optimizer = optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay)

    # Initialize trainer, validator and plotter
    trainer = Trainer(model, dataset, loss_measurer, optimizer)
    validator = Validator(model, dataset, loss_measurer)
    plotter = MetricsPlotter()

    # Tran and validate
    for epoch in range(max_epochs):
        train_loss = trainer.fit()
        validate_loss, perplexity = validator()
        scores = evaluate(dataset, model)
        plotter.add(epoch, train_loss, validate_loss, perplexity)

        logger.info("epoch #{}, train_loss = {:.3f}, validate_loss = {:.3f}, perplexity = {:.2f}, "
                    "bleu_scores = {}",
                    epoch, train_loss, validate_loss, perplexity,
                    [f'{s:.2f}' for s in scores])

    logger.info("device = {}, elapsed time: {:.1f} seconds", device, time.time() - start_sec)

    # Plot metrics
    plotter.plot(title=f"Machine Translation ({AttentionSeq2SeqTranslator.__name__})")

    # Plot attention weights
    plot_attention_weights_heatmap(model.attention_weights)

    # Evaluate
    evaluate(dataset, model, print_log=True)

    logger.info("done!")


if __name__ == "__main__":
    logger.remove()
    logger.add(sink=sys.stderr, level="INFO")

    main()

    # epoch #29, train_loss = 0.878, validate_loss = 4.159, perplexity = 64.00, bleu_scores = ['0.00', '1.00', '0.00', '0.69']
    # device = cuda, elapsed time: 303.0 seconds
    # prediction: #0
    #     source = 'have you been eating ?'
    #     target = '你在吃东西吗？'
    #     prediction = '你吃過东京嗎？'
    #     bleu = 0.00
    # prediction: #1
    #     source = 'tom has always been a bachelor .'
    #     target = '汤姆一直都是单身汉。'
    #     prediction = '汤姆一直都是单身汉。'
    #     bleu = 1.00
    # prediction: #2
    #     source = 'with the t .v . on , how can you keep your mind on your studies ?'
    #     target = '你开着电视怎么能安心学习呢？'
    #     prediction = '请把你的姓法能放在学院里放作。'
    #     bleu = 0.00
    # prediction: #3
    #     source = 'i didn't believe him at first .'
    #     target = '起初我不相信他。'
    #     prediction = '我不相信他在那里。'
    #     bleu = 0.69
    # done!
