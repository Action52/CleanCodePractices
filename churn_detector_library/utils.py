"""
This library contains the functions necessary to implement the churn models,
in particular using LogisticRegression and RandomForests.
"""

import os
from typing import Any, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve
from sklearn.model_selection import GridSearchCV, train_test_split

sns.set()

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
current_file_path = os.path.abspath(__file__)
current_directory = os.path.dirname(current_file_path)
base_path = os.path.join(current_directory, "..")


def save_plot(filepath: str, title: str, plot_function: Any, **kwargs) -> None:
    """
    Helper function to save a plot.
    Args:
        filepath: The path in which to save the plot
        title: The title for the plot
        plot_function: Plot function to call. i.e: df['Customer_Age'].hist

    Returns: None
    """
    plt.title(title)
    plot_function(**kwargs)
    plt.savefig(filepath)


def import_data(path: str) -> pd.DataFrame:
    """ Reads a path string pointing to a csv, returns a pandas dataframe.

    Args:
        path (str): The path to the csv.
    Returns:
        pd.Dataframe: The pandas dataframe representing the csv.
    """
    df = pd.read_csv(path, index_col=0)
    return df


def perform_eda(df: pd.DataFrame, config: dict) -> None:
    """
    Perform eda on df: Logs basic tables and saves figures to images folder.
    Args:
        df (pd.DataFrame): The csv pandas dataframe.
        config (dict): The loaded config file.
    Returns: None
    """

    # Create churn column
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Plot and save figures
    plt.figure(figsize=(20, 10))

    images_path = f"{base_path}/images/eda/"

    # Churn histogram
    save_plot(
        f"{images_path}{config['eda']['churn_histogram']}",
        "Churn Histogram",
        plot_function=df["Churn"].hist
    )

    # Customer age histogram
    save_plot(
        f"{images_path}{config['eda']['customer_age']}",
        "Customer Age Histogram",
        plot_function=df["Customer_Age"].hist
    )

    # Marital status histogram
    save_plot(
        f"{images_path}{config['eda']['marital_status']}",
        "Normalized Marital Status ",
        plot_function=df.Marital_Status.value_counts('normalize').plot,
        kind="bar"
    )

    # Distributions of 'Total_Trans_Ct' adding a smooth curve using a
    # kernel density estimate
    save_plot(
        f"{images_path}{config['eda']['trans_count']}",
        "Total Trans Density Count",
        sns.histplot,
        data=df["Total_Trans_Ct"],
        stat='density',
        kde=True
    )

    # Correlation Heatmap
    save_plot(
        f"{images_path}{config['eda']['correlations']}",
        "Column Correlation Heatmap",
        sns.heatmap,
        data=df.corr(),
        annot=False,
        cmap='Dark2_r',
        linewidths=2
    )


def encoder_helper(
        df: pd.DataFrame,
        category_lst: list,
        response: Optional[str] = '_Churn') -> pd.DataFrame:
    """
    Helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the
    notebook.
    Args:
        df: pandas dataframe
        category_lst: list of columns with categorical features
        response: string of response name (optional argument that could be used)
            for naming variables or index y column. Default: _Churn

    Returns: A pd.DataFrame

    """
    for category in category_lst:
        category_grouped = df.groupby(category).mean()['Churn']
        df[f'{category}{response}'] = df[category].map(category_grouped)
    return df


def perform_feature_engineering(df: pd.DataFrame,
                                config: dict,
                                response: str = "_Churn") -> Tuple[pd.DataFrame,
                                                                   pd.DataFrame,
                                                                   pd.Series,
                                                                   pd.Series]:
    """
    Performs the feature engineering steps on a churn dataset df.
    Args:
        df: Pandas dataframe
        config: configuration dict
        response: string of response name [optional argument that could be used
        for naming variables or index y column]. Default: "_Churn"

    Returns: Tuple of:
            X_train: X training data
            X_test: X testing data
            y_train: y training data
            y_test: y testing data

    """
    categories = config['cat_columns']
    columns_to_keep = config['keep_cols']
    encoded_df = encoder_helper(df, categories, response)
    y = df['Churn']
    X = pd.DataFrame()
    X[columns_to_keep] = encoded_df[columns_to_keep]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42)
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train: pd.Series,
                                y_test: pd.Series,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf,
                                config: dict) -> None:
    '''
    produces classification report for training and testing results and stores
    report as image in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
            config: the config dict.
    output:
             None
    '''
    # Map the coordinates and texts to avoid repetitive code
    text_args = config['report']['text_args']
    fontdict = text_args['fontdict']
    fontproperties = text_args['fontproperties']
    y_coordinates = [1.25, 0.05, 0.6, 0.7]
    save_paths = config['report']['save_file']
    model_texts = [
        [
            "Random Forest Train",
            str(classification_report(y_test, y_test_preds_rf)),
            "Random Forest Test ",
            str(classification_report(y_train, y_train_preds_rf))
        ],
        [
            "Logistic Regression Train",
            str(classification_report(y_train, y_train_preds_lr)),
            "Logistic Regression Test",
            str(classification_report(y_test, y_test_preds_lr))
        ]
    ]

    for model_text, save_path in zip(model_texts, save_paths):
        plt.figure(figsize=(10, 5))
        plt.rc('figure', figsize=(10, 5))
        for y, text in zip(y_coordinates, model_text):
            plt.text(
                x=0.01,
                y=y,
                s=text,
                fontdict=fontdict,
                fontproperties=fontproperties)
        plt.axis('off')
        filepath = config['report']['save_file'][save_path]
        plt.savefig(filepath)


def feature_importance_plot(
        model,
        X_data: pd.DataFrame,
        output_pth: str) -> None:
    """
    Creates and stores the feature importances plot.
    Args:
        model: Model with the feature importances
        X_data: Pandas dataframe to process.
        output_pth: Path to save the plot.

    Returns: None

    """
    # Calculate feature importance
    importances = model.feature_importances_
    # Sort feature importance in descending order
    indices = np.argsort(importances)[::-1]
    # Rearrange feature names so they match the sorted feature importance
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)


def train_models(
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
        config: dict) -> None:
    """
    Train, store model results: images + scores, and store models
    Args:
        X_train: The training data X pandas dataframe.
        X_test: The testing data X pandas dataframe.
        y_train: The training data y pandas series.
        y_test: The testing data y pandas series.
        config: Configuration dictionary.

    Returns: None

    """
    # Extract the args from the config dictionary
    forest_args = config['train_model']['random_forest']['args']
    log_regression_args = config['train_model']['logistic_regression']['args']
    grid_search_args = config['train_model']['random_forest']['grid_search']

    # Instantiate the classifiers
    rfc = RandomForestClassifier(**forest_args)
    lrc = LogisticRegression(**log_regression_args)

    # Use grid search to fit random forest
    cv_rfc = GridSearchCV(estimator=rfc, **grid_search_args)
    cv_rfc.fit(X_train, y_train)

    # Fit the logistic regression classifier
    lrc.fit(X_train, y_train)

    # Predict
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Store the models
    forest_path = config['train_model']['random_forest']['save_file']
    logistic_path = config['train_model']['logistic_regression']['save_file']
    joblib.dump(cv_rfc.best_estimator_, forest_path)
    joblib.dump(lrc, logistic_path)

    # Store the results
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf,
        config)

    # Create and store the ROC curves
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(
        cv_rfc.best_estimator_,
        X_test,
        y_test,
        ax=ax,
        alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    roc_path = config['results']['roc_curves']
    plt.savefig(f'{base_path}/images/results/{roc_path}')

    # Create and store the feature importances
    importances_path = config['results']['feature_importance']
    feature_importance_pth = f"{base_path}/images/results/{importances_path}"

    feature_importance_plot(
        cv_rfc.best_estimator_,
        X_test,
        feature_importance_pth)
