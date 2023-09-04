"""
This script triggers the test for the model and logs the results.
"""

# Imports

import logging
import os

import pandas as pd
import pytest

from churn_detector_library.utils import (encoder_helper, import_data,
                                          perform_eda,
                                          perform_feature_engineering,
                                          train_models)

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
base_path = os.path.join(current_directory, "")


@pytest.mark.parametrize(
    "path",
    ["./data/bank_data.csv"])
def test_import(path: str) -> None:
    """ Test to check that the import_data function works as expected.

    Args:
            path (str): String to test
    Raises:
            err: FileNotFoundError if file is not located.
            err: AssertionError if dataframe shape is not as expected.
    """
    try:
        logging.info("Beginning test imports.")
        df = import_data(path)
        pytest.df = df
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_data: The file wasn't found.")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and "
            "columns")
        raise err


def test_eda() -> None:
    """
    Tests the exploratory data analysis function.
    Returns: None
    """
    logging.info("Beginning test eda.")
    df: pd.DataFrame = pytest.df
    config: dict = pytest.churn_config
    categories = config["cat_columns"] + config["quant_columns"]
    perform_eda(df, config)
    # Verify the columns exist
    for category in categories:
        try:
            assert category in df.columns
        except AssertionError as err:
            logging.error(f"Category {category} not found in df.")
            raise err

    images_path = f"{base_path}/images/eda/"
    # Check that the plots were generated
    for eda in config['eda']:
        plot_path = config['eda'][eda]
        try:
            assert os.path.exists(f"{images_path}{plot_path}")
        except AssertionError as err:
            logging.error(f"{images_path}{plot_path} not found.")
            raise err
    logging.info("Test eda: SUCCESS")


@pytest.mark.parametrize(
    "categories",
    [
        pytest.churn_config['cat_columns']
    ])
def test_encoder_helper(categories: list) -> None:
    """
    Tests the encoder helper function.
    Args:
            categories: list of categories to pass to the function.
    Returns: None
    """
    logging.info("Beginning test encoder helper.")
    df: pd.DataFrame = pytest.df
    encoded_df = encoder_helper(df, categories, response="_Churn")
    for category in categories:
        try:
            assert f'{category}_Churn' in encoded_df
        except AssertionError as err:
            logging.error(f"Encoded category {category}_Churn not found.")
            raise err
    pytest.encoded_df = encoded_df
    logging.info("Test encoder helper: SUCCESS.")


def test_perform_feature_engineering() -> None:
    """
    Tests the feature engineering function.
    Returns: None
    """
    logging.info("Beginning test perform feature engineering.")
    encoded_df = pytest.df
    config = pytest.churn_config
    cols_to_keep = config["keep_cols"]
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        encoded_df, config)
    try:
        assert isinstance(y_train, pd.Series)
        assert isinstance(y_test, pd.Series)
        assert isinstance(x_train, pd.DataFrame)
        assert isinstance(x_test, pd.DataFrame)
        assert x_train.shape[0] == len(y_train)
        assert x_test.shape[0] == len(y_test)
    except AssertionError as err:
        logging.error("The shape of one of the resulting dataframes is wrong.")
        raise err
    for category in cols_to_keep:
        try:
            assert category in x_train.columns
            assert category in x_test.columns
        except AssertionError as err:
            logging.error(
                f"{category} is not in both X_train and X_test columns")
            raise err
    pytest.train_test_split_dfs = (x_train, x_test, y_train, y_test)
    logging.info("Test perform feature engineering: SUCCESS.")


def test_train_models():
    '''
    test train_models
    '''
    logging.info("Beginning test train models.")
    x_train, x_test, y_train, y_test = pytest.train_test_split_dfs
    config = pytest.churn_config
    train_models(x_train, x_test, y_train, y_test, config)
    images_path = f"{base_path}/images/results/"
    # Check that the plots and report were generated
    for result in config['results']:
        plot_path = config['results'][result]
        try:
            assert os.path.exists(f"{images_path}{plot_path}")
        except AssertionError as err:
            logging.error(f"{images_path}{plot_path} not found.")
            raise err
    logging.info("Test train models: SUCCESS.")
