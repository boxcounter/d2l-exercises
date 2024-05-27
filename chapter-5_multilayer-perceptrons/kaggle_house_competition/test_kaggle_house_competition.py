# pylint:disable=missing-function-docstring, missing-module-docstring, missing-class-docstring
# pylint:disable=too-few-public-methods, too-many-arguments
# pylint:disable=import-outside-toplevel
# pylint:disable=too-many-locals
# pylint:disable=protected-access,redefined-outer-name

import pandas as pd
import pytest
import torch
import torch.utils
import torch.utils.data
from kaggle_house_competition import KaggleHouseDateset


def test_fold_data():
    df = pd.DataFrame({
        'Feature1': list(range(0, 11)),
        'Feature2': list(range(11, 22)),
        'SalePrice': list(range(100, 1200, 100)),
    })
    num_folds = 5
    folds = KaggleHouseDateset._fold_data(df, num_folds)

    assert len(folds) == num_folds
    for i, fold in enumerate(folds):
        assert isinstance(fold, tuple)
        assert len(fold) == 2

        for tensor in fold:
            assert isinstance(tensor, torch.Tensor)

            total_rows = df.shape[0]
            if i == num_folds - 1:
                assert tensor.shape[0] == total_rows // num_folds + total_rows % num_folds
            else:
                assert tensor.shape[0] == total_rows // num_folds


def test_preprocess_data():
    NAN = float('nan')
    train_df = pd.DataFrame({
        'Id': [1, 2, 3, 4, 5, 6],
        'Feature1': [0., 0., 0., 0., 0., 0.],
        'Feature2': [1, NAN, 3, 4, NAN, 6],
        'Feature3': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature4': ['male', 'male', 'female', 'female', 'female', 'male'],
        'SalePrice': [100, 200, 300, 400, 500, 600],
    })

    test_df = pd.DataFrame({
        'Id': [1, 2, 3, 4, 5, 6],
        'Feature1': [0., 0., 0., 0., 0., 0.],
        'Feature2': [1, NAN, 3, 4, NAN, 6],
        'Feature3': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature4': ['male', 'male', 'female', 'unknown', 'female', 'male'],
    })

    expected_preprocessed_train_df = pd.DataFrame({
        'Feature1': [0., 0., 0., 0., 0., 0.],
        'Feature2': [-1.29, 0., -0.25, 0.25, 0., 1.29],
        'Feature3': [0., 0., 0., 0., 0., 0.],
        'Feature4_male': [1, 1, 0, 0, 0, 1],
        'Feature4_female': [0, 0, 1, 1, 1, 0],
        'Feature4_unknown': [0, 0, 0, 0, 0, 0],
        'SalePrice': [100, 200, 300, 400, 500, 600],
    })

    expected_preprocessed_test_df = pd.DataFrame({
        'Feature1': [0., 0., 0., 0., 0., 0.],
        'Feature2': [-1.29, 0., -0.25, 0.25, 0., 1.29],
        'Feature3': [0., 0., 0., 0., 0., 0.],
        'Feature4_male': [1, 1, 0, 0, 0, 1],
        'Feature4_female': [0, 0, 1, 0, 1, 0],
        'Feature4_unknown': [0, 0, 0, 1, 0, 0],
    })

    preprocessed_train_df, preprocessed_test_df = KaggleHouseDateset._preprocess_data(
        train_df, test_df)

    pd.testing.assert_frame_equal(
        preprocessed_train_df, expected_preprocessed_train_df, rtol=0.1, check_like=True)
    pd.testing.assert_frame_equal(
        preprocessed_test_df, expected_preprocessed_test_df, rtol=0.1, check_like=True)


@pytest.mark.parametrize(
        ("num_folds", "expected_lens"),
        [
            (5, [(5,1), (4,2)]),
            (3, [(4,2)]),
            (2, [(3,3)])
        ]
    )
def test_kaggle_house_dataset(
    num_folds: int,
    expected_lens: list[tuple[int, int]],
):
    NAN = float('nan')
    train_df = pd.DataFrame({
        'Id': [1, 2, 3, 4, 5, 6],
        'Feature1': [0., 0., 0., 0., 0., 0.],
        'Feature2': [1, NAN, 3, 4, NAN, 6],
        'Feature3': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature4': ['male', 'male', 'female', 'unknown', 'female', 'male'],
        'Feature5': [0., 0., 0., 0., 0., 0.],
        'Feature6': [1, NAN, 3, 4, NAN, 6],
        'Feature7': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature8': ['male', 'male', 'female', 'unknown', 'female', 'male'],
        'Feature9': [0., 0., 0., 0., 0., 0.],
        'Feature10': [1, NAN, 3, 4, NAN, 6],
        'Feature11': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature12': ['male', 'male', 'female', 'unknown', 'female', 'male'],
        'SalePrice': [100, 200, 300, 400, 500, 600],
        })

    test_df = pd.DataFrame({
        'Id': [1, 2, 3, 4, 5, 6],
        'Feature1': [0., 0., 0., 0., 0., 0.],
        'Feature2': [1, NAN, 3, 4, NAN, 6],
        'Feature3': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature4': ['male', 'male', 'female', 'unknown', 'female', 'male'],
        'Feature5': [0., 0., 0., 0., 0., 0.],
        'Feature6': [1, NAN, 3, 4, NAN, 6],
        'Feature7': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature8': ['male', 'male', 'female', 'unknown', 'female', 'male'],
        'Feature9': [0., 0., 0., 0., 0., 0.],
        'Feature10': [1, NAN, 3, 4, NAN, 6],
        'Feature11': [NAN, NAN, NAN, NAN, NAN, NAN],
        'Feature12': ['male', 'male', 'female', 'unknown', 'female', 'male'],
        })

    dataset = KaggleHouseDateset(train_df, test_df, num_folds)
    assert dataset._train_folds is not None
    assert len(dataset._train_folds) == num_folds
    assert isinstance(dataset._train_folds, list)

    assert dataset._test_data is not None
    assert isinstance(dataset._test_data, torch.Tensor)

    for i in range(num_folds):
        train_loader, valid_loader = dataset.get_data_loaders(i, batch_size=1)

        assert isinstance(train_loader, torch.utils.data.DataLoader)
        assert isinstance(valid_loader, torch.utils.data.DataLoader)

        assert (len(train_loader), len(valid_loader)) in expected_lens
