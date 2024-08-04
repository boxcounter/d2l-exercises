# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments, too-many-instance-attributes
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals
# pylint:disable=protected-access
import math

import pytest

from machine_translation import TextSequenceDataset, BLEU


def test_preprocess_source_text_sequence():
    raw = "Hello,world! This is an example:Python is great.Are you learning?"
    expected = "hello ,world ! this is an example:python is great .are you learning ?"

    text = TextSequenceDataset.preprocess_source_text_sequence(raw)
    for x, y in zip(text, expected.split(), strict=True):
        assert x == y


def test_extract_lines_of_words(tmp_path):
    raw_content = """
Hi.	嗨。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #891077 (Martha)
Hi.	你好。	CC-BY 2.0 (France) Attribution: tatoeba.org #538123 (CM) & #4857568 (musclegirlxyp)
Run.	你用跑的。	CC-BY 2.0 (France) Attribution: tatoeba.org #4008918 (JSakuragi) & #3748344 (egg0073)
Stop!	住手！	CC-BY 2.0 (France) Attribution: tatoeba.org #448320 (CM) & #448321 (GlossaMatik)
Wait!	等等！	CC-BY 2.0 (France) Attribution: tatoeba.org #1744314 (belgavox) & #4970122 (wzhd)
Wait!	等一下！	CC-BY 2.0 (France) Attribution: tatoeba.org #1744314 (belgavox) & #5092613 (mirrorvan)
Begin.	开始！	CC-BY 2.0 (France) Attribution: tatoeba.org #6102432 (mailohilohi) & #5094852 (Jin_Dehong)
Hello!	你好。	CC-BY 2.0 (France) Attribution: tatoeba.org #373330 (CK) & #4857568 (musclegirlxyp)
I try.	我试试。	CC-BY 2.0 (France) Attribution: tatoeba.org #20776 (CK) & #8870261 (will66)
I won!	我赢了。	CC-BY 2.0 (France) Attribution: tatoeba.org #2005192 (CK) & #5102367 (mirrorvan)
Oh no!	不会吧。	CC-BY 2.0 (France) Attribution: tatoeba.org #1299275 (CK) & #5092475 (mirrorvan)
Cheers!	乾杯!	CC-BY 2.0 (France) Attribution: tatoeba.org #487006 (human600) & #765577 (Martha)
"""

    source = """
hi .
hi .
run .
stop !
wait !
wait !
begin .
hello !
i try .
i won !
oh no !
cheers !
""".strip().split('\n')

    target = """
嗨。
你好。
你用跑的。
住手！
等等！
等一下！
开始！
你好。
我试试。
我赢了。
不会吧。
乾杯!
""".strip().split('\n')

    tmp_file = tmp_path / "tsv.tmp"
    with open(tmp_file, 'w', encoding='UTF-8') as f:
        f.write(raw_content)

    source_corpus, target_corpus = TextSequenceDataset._extract_lines_of_words(tmp_file)

    for x, y in zip(source_corpus, source, strict=True):
        assert x, y

    for x, y in zip(target_corpus, target, strict=True):
        assert x, y


class TestBLEU:
    @pytest.fixture
    def sequences(self) -> tuple[list[str], list[str]]: # (prediction_words, target_words)
        return (['A', 'B', 'B', 'C', 'D'], ['A', 'B', 'C', 'D', 'E', 'F'])

    def test_ngram(self, sequences):
        prediction_words, _ = sequences

        ngrams = BLEU._ngram(prediction_words, 1)
        expected = [('A',), ('B',), ('B',), ('C',), ('D',)]
        assert ngrams == expected

        ngrams = BLEU._ngram(prediction_words, 2)
        expected = [('A', 'B'), ('B', 'B'), ('B', 'C'), ('C', 'D')]
        assert ngrams == expected

        ngrams = BLEU._ngram(prediction_words, 3)
        expected = [('A', 'B', 'B'), ('B', 'B', 'C'), ('B', 'C', 'D')]
        assert ngrams == expected

        ngrams = BLEU._ngram(prediction_words, 4)
        expected = [('A', 'B', 'B', 'C'), ('B', 'B', 'C', 'D')]
        assert ngrams == expected

    def test_precision(self, sequences):
        prediction_words, target_words = sequences

        precision = BLEU._precision(prediction_words, target_words, 1)
        assert math.isclose(precision, 4/5)

        precision = BLEU._precision(prediction_words, target_words, 2)
        assert math.isclose(precision, 3/4)

        precision = BLEU._precision(prediction_words, target_words, 3)
        assert math.isclose(precision, 1/3)

        precision = BLEU._precision(prediction_words, target_words, 4)
        assert math.isclose(precision, 0.0)

    def test_score(self):
        assert math.isclose(1.0, BLEU('ABCDE', 'ABCDE').score)
        assert math.isclose(0.0, BLEU('XYZ', 'ABCDE').score)
