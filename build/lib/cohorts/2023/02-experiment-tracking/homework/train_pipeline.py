"""train_pipeline.py """

import os
import pickle
import click
import pandas as pd
import yaml
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor


def read_dataframe(filename: str):
    df = pd.read_parquet(filename)
    return df


def create_pipeline() -> Pipeline:
    """Function to create sklearn pipeline"""

    return pipeline


# @click.command()
# @click.option(
#     "--data_config",
#     help="Path to the YAML configuration file"
# )
# @click.option(
#     "--dest_path",
#     help="Location where the resulting files will be saved"
# )
def run_data_prep(data_config: str, dest_path: str):
    """
    Runs data preparation steps based on the provided configuration file.

    Args:
        data_config (str): The file path of the data configuration file.
        dest_path (str): The destination path to save the processed data.

    Returns:
        None
    """

    with open(data_config, "r", encoding="utf-8") as filename:
        config = yaml.safe_load(filename)

    train_path = config["train"]["path"]
    train_dataset = config["train"].get("dataset")
    val_path = config["val"]["path"]
    val_dataset = config["val"].get("dataset")
    test_path = config["test"]["path"]
    test_dataset = config["test"].get("dataset")

    # Load parquet files
    df_train = read_dataframe(os.path.join(train_path, f"{train_dataset}"))
    df_val = read_dataframe(os.path.join(val_path, f"{val_dataset}"))
    df_test = read_dataframe(os.path.join(test_path, f"{test_dataset}"))

    # Extract the target
    target = "tip_amount"
    y_train = df_train[target].values
    y_val = df_val[target].values
    y_test = df_test[target].values

    # Create pipeline and fit on training data
    pipeline = create_pipeline()
    pipeline.fit(df_train.iloc[:, 3:], y_train)

    # # Transform data using the pipeline
    # X_train = pipeline.transform(df_train)
    # X_val = pipeline.transform(df_val)
    # X_test = pipeline.transform(df_test)

    # # Create dest_path folder unless it already exists
    # os.makedirs(dest_path, exist_ok=True)

    # # Save pipeline and datasets
    # dump_pickle(pipeline, os.path.join(dest_path, "pipeline.pkl"))
    # dump_pickle((X_train, y_train), os.path.join(dest_path, "train.pkl"))
    # dump_pickle((X_val, y_val), os.path.join(dest_path, "val.pkl"))
    # dump_pickle((X_test, y_test), os.path.join(dest_path, "test.pkl"))


if __name__ == "__main__":
    # run_data_prep()
    run_data_prep(
        data_config="./cohorts/2023/02-experiment-tracking/homework/data_config.yaml",
        dest_path="./cohorts/2023/02-experiment-tracking/homework/artifacts",
    )
