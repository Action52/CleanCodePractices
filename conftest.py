"""
    This file handles the pytest configurations.
"""

import logging
import os

import pytest
import yaml

current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
base_path = os.path.join(current_directory, "")


def load_config(path: str) -> dict:
    """
    Loads the configuration file in yaml format.

    Args:
        path (str): The path to the configuration file. i.e: "~/config.yaml"

    Returns:
        dict: Dict with the keys as depicted in the yaml configuration file.
    """
    try:
        with open(path, 'r') as config_file:
            config_data = yaml.safe_load(config_file)
            return config_data
    except FileNotFoundError as err:
        logging.error(
            f"ERROR: {err}. The yaml config file was not found in "
            f"this path: {path}")
    except yaml.YAMLError as err:
        logging.error(
            f"ERROR: {err}. Something happened while parsing the config file.")


def load_logger(config: dict) -> None:
    """Sets a simple log with the specified configuration.

    Args:
        config (dict): The loaded config yaml dict.
    """
    logging_config = config.get("logging")
    log_filename = logging_config.get("log_filename")
    log_level = logging_config.get("log_level")
    log_mode = logging_config.get("log_mode")
    log_format = logging_config.get("log_format")

    logging.basicConfig(
        filename=log_filename,
        level=log_level,
        filemode=log_mode,
        format=log_format
    )


def config(config_path: str = "config.yaml") -> dict:
    """ Sets up the configuration when testing.
    Args:
        config_path (str): The path to the configuration file.
    Returns:
        dict: A dict with the configuration.
    """
    config_path = f"{base_path}config.yaml"
    return load_config(config_path)


def logger(config: dict) -> None:
    """ Loads the logger.

    Args:
        config (dict): The configuration dictionary.
    """
    load_logger(config)


def df_plugin():
    """ Plugin for the dataframe during tests."""
    return 0


def encoded_df_plugin():
    """ Plugin for the encoded dataframe during tests."""
    return 0


def train_test_split_df_plugin():
    """ Plugin for the X, y, train, test dataframes during tests."""
    return 0


def pytest_configure():
    """
    Configures the pytest environment.
    """
    folders = [
        "images"
        "images/eda",
        "images/results",
        "logs",
        "models"
    ]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
        else:
            pass
    pytest.df = df_plugin()
    pytest.encoded_df = encoded_df_plugin()
    pytest.churn_config = config()
    pytest.train_test_split_dfs = train_test_split_df_plugin()
    logger(pytest.churn_config)
