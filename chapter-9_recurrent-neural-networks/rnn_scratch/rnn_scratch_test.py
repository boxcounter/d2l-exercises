# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments
# pylint:disable=import-outside-toplevel
# pylint:disable=protected-access
# pylint:disable=redefined-outer-name
import pytest

from rnn_scratch import TimeMachineDataset, Vocabulary


@pytest.fixture
def tokens() -> list[str]:
    return [
        'red', 'blue', 'black', 'green', 'yellow', 'gray',
        'red', 'blue', 'black', 'green', 'yellow',
        'red', 'blue', 'black', 'green',
        'red', 'blue', 'black',
        'red', 'blue',
        'red',
    ]


class TestVocabulary:
    def test_default(
        self,
        tokens: list[str]
    ) -> None:
        vocab = Vocabulary(tokens)
        assert vocab._tokens_freq == {
            '<unk>': 0, 'red': 6, 'blue': 5, 'black': 4, 'green': 3, 'yellow': 2, 'gray': 1
        }

        assert vocab.most_common(3) == [
            ('red', 6), ('blue', 5), ('black', 4)
        ]

        assert len(vocab) == len(set(tokens)) + len(vocab.RESERVED_TOKENS)

    def test_min_freq(
        self,
        tokens: list[str],
    ) -> None:
        vocab = Vocabulary(tokens, min_freq=3)
        expected_tokens_freq = {
            'red': 6, 'blue': 5, 'black': 4, 'green': 3, '<unk>': 0
        }

        assert vocab._tokens_freq == expected_tokens_freq
        assert len(vocab) == len(expected_tokens_freq)

    def test_reserved_tokens(
        self,
        tokens: list[str],
    ) -> None:
        reserved_tokens=('<pad>', '<bos>', '<eos>')
        vocab = Vocabulary(tokens, reserved_tokens=reserved_tokens)
        assert vocab._tokens_freq == {
            '<pad>': 0, '<bos>': 0, '<eos>': 0, '<unk>': 0,
            'red': 6, 'blue': 5, 'black': 4, 'green': 3, 'yellow': 2, 'gray': 1
        }

        expected = len(set(tokens)) + len(vocab.RESERVED_TOKENS) + len(reserved_tokens)
        assert len(vocab) == expected


class TestTimeMachineDataset:
    def test_preprocess_tokens(self) -> None:
        content = list("abcdefghijklmnopqrstuvwxyz")
        vocab = Vocabulary(content)
        data = vocab.tokenize(list(content))

        num_steps = [3, 5, 8]
        for n in num_steps:
            X, Y = TimeMachineDataset._preprocess_tokens(data, n)
            assert X.shape == (len(data)-n, n)
            assert X[0].tolist() == data[:n]
            assert Y.shape == (len(data)-n, n)
            assert Y[0].tolist() == data[1:n+1]
