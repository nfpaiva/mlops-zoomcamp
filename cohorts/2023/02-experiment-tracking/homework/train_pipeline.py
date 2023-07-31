"""train_pipeline.py """
import argparse
import datetime
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Union
from dataclasses import dataclass

import pandas as pd
import numpy as np
import yaml

import optuna
import mlflow  # type: ignore
from mlflow.tracking import MlflowClient  # type: ignore
from optuna.samplers import TPESampler
from sklearn.ensemble import RandomForestRegressor  # type: ignore
from sklearn.feature_extraction import DictVectorizer  # type: ignore
from sklearn.metrics import mean_squared_error  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """A data class representing the split datasets and labels.

    This data class encapsulates the training, validation, and testing datasets along with their
    corresponding labels. It is designed to hold the following data:

    Attributes:
        x_train (pd.DataFrame): The pd DataFrame containing the features of the training dataset.
        x_val (pd.DataFrame): The pd DataFrame containing the features of the validation dataset.
        x_test (pd.DataFrame): The pd DataFrame containing the features of the testing dataset.
        y_train (np.ndarray): The NumPy array containing the labels of the training dataset.
        y_val (np.ndarray): The NumPy array containing the labels of the validation dataset.
        y_test (np.ndarray): The NumPy array containing the labels of the testing dataset.

    Example:
        # Create a DatasetSplit object
        dataset_split_opt = DatasetSplit(
            x_train=train_features,
            x_val=val_features,
            x_test=test_features,
            y_train=train_labels,
            y_val=val_labels,
            y_test=test_labels,
        )

        # Access the individual datasets and labels
        x_train_data = dataset_split_opt.x_train
        y_test_labels = dataset_split_opt.y_test
    """

    x_train: pd.DataFrame
    x_val: pd.DataFrame
    x_test: pd.DataFrame
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


class DataHandler:
    """A class to handle data loading, preprocessing, and splitting.

    This class provides methods to download data from an S3 bucket, load it into pandas DataFrames,
    preprocess the dataset by applying transformations, and split the data into
    training, validation, and testing sets. The class encapsulates data handling operations
    and allows for easy management of the data required for training and evaluation.

    Attributes:
        data_dir (Path): The directory where the data will be saved.
        s3_url (str): The S3 URL of the data to be downloaded.

    Methods:
        download_data(parquet_file: str) -> Path:
            Download data from S3 and save it to the local directory.

        load_dataset(file_names: Dict[str, Dict[str, str]]) -> Dict[str, pd.DataFrame]:
            Load the dataset from local files.

        preprocess_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
            Preprocess the input dataset.

        load_split_dataset() -> DatasetSplit:
            Load and preprocess the dataset into training, validation, and testing splits.

    """

    def __init__(self, data_dir: Path, s3_url: str) -> None:
        """Constructor for the DataHandler class.

        Args:
            data_dir (Path): The directory where the data will be saved.
            s3_url (str): The S3 URL of the data to be downloaded.
        """
        self.data_dir = data_dir
        self.s3_url = s3_url

    def download_data(self, parquet_file: str) -> Path:
        """Download data from S3 and save it to the local directory.

        Args:
            parquet_file (str): Name of the file to be downloaded.

        Returns:
            Path: Path to the downloaded file.
        """
        file_path = self.data_dir / parquet_file
        url = self.s3_url + parquet_file
        logger.info("Attempting to download filename %s", url)
        if not os.path.isfile(file_path):
            logger.info("File does not exist, downloading from S3 bucket.")
            if not os.path.exists(os.path.dirname(file_path)):
                os.makedirs(os.path.dirname(file_path))
            subprocess.run(["wget", "-O", file_path, url], check=True)
            logger.info("File downloaded successfully and saved at %s", file_path)
        else:
            logger.info("File already exists.")
        return file_path

    def preprocess_dataset(self, dataset: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the input dataset.

        This method performs preprocessing on the input dataset, including the following steps:
        1. Calculate the trip duration by subtracting the pickup datetime from the dropoff datetime,
        and then convert it to minutes.
        2. Filter out trips with durations less than 1 minute or greater than 60 minutes.
        3. Convert selected columns ("PULocationID" and "DOLocationID") to string data types.
        4. Create a new categorical feature "PU_DO" by combining "PULocationID" and "DOLocationID".
        5. Ensure that the "PU_DO" column is of type string.
        6. Select and return only the relevant categorical ("PU_DO") and numerical ("trip_distance",
        "tip_amount") columns required for modeling.

        Args:
            dataset (pd.DataFrame): The input dataset to be preprocessed.

        Returns:
            pd.DataFrame: The preprocessed dataset containing the following columns:
                        - 'PU_DO': A categorical feature representing the combined
                            Pickup Location ID and Dropoff Location ID.
                        - 'trip_distance': A numerical feature representing the
                            distance of the trip.
                        - 'tip_amount': A numerical feature representing the tip amount
                            given by the passenger.
        """
        dataset["duration"] = (
            dataset["lpep_dropoff_datetime"] - dataset["lpep_pickup_datetime"]
        ).dt.total_seconds() / 60
        filtered_dataset = dataset[
            (dataset.duration >= 1) & (dataset.duration <= 60)
        ].copy()
        str_columns = ["PULocationID", "DOLocationID"]
        filtered_dataset.loc[:, str_columns] = filtered_dataset[str_columns].astype(str)
        filtered_dataset.loc[:, "PU_DO"] = (
            filtered_dataset["PULocationID"] + "_" + filtered_dataset["DOLocationID"]
        )
        filtered_dataset.loc[:, "PU_DO"] = filtered_dataset["PU_DO"].astype(str)
        categorical = ["PU_DO"]
        numerical = ["trip_distance", "tip_amount"]

        return filtered_dataset[categorical + numerical]

    def load_dataset(
        self, file_names_load: Dict[str, Dict[str, str]]
    ) -> Dict[str, pd.DataFrame]:
        """Load the dataset from local files.

        Args:
            file_names (Dict[str, Dict[str, str]]): Dictionary containing file
                names and their corresponding URLs.

        Returns:
            Dict[str, pd.DataFrame]: A dictionary containing datasets loaded from local files,
                with keys corresponding to the file types (e.g., "train", "val", "test").
        """
        df_data = {}
        for dataset, inner_month_info in file_names_load.items():
            file_path = self.download_data(inner_month_info["file_name"])
            df_data[dataset] = pd.read_parquet(file_path)

        return df_data

    def load_split_dataset(self, file_names: Dict[str, Dict[str, str]]) -> DatasetSplit:
        """Load and preprocess the dataset into training, validation, and testing splits.

        This method downloads and loads the dataset from the specified file names and then
        preprocesses it. The preprocessing steps include calculating the trip duration,
        filtering out trips with durations less than 1 minute or greater than 60 minutes,
        and converting selected columns to string data types. The preprocessed dataset
        contains only the relevant categorical and numerical columns required for modeling.

        Returns:
            DatasetSplit: A data class representing the split datasets and labels as follows:
                        - x_train: A pandas DataFrame containing the training data features.
                        - x_val: A pandas DataFrame containing the validation data features.
                        - x_test: A pandas DataFrame containing the testing data features.
                        - y_train: A NumPy array containing the training data labels.
                        - y_val: A NumPy array containing the validation data labels.
                        - y_test: A NumPy array containing the testing data labels.
        """

        df_data = self.load_dataset(file_names)

        df_data_processed = {
            dataset: self.preprocess_dataset(df_data[dataset])
            for dataset in file_names.keys()
        }

        x_train_inner = df_data_processed["train"]
        x_val_inner = df_data_processed["val"]
        x_test_inner = df_data_processed["test"]

        y_train_inner = df_data_processed["train"]["tip_amount"]
        y_val_inner = df_data_processed["val"]["tip_amount"]
        y_test_inner = df_data_processed["test"]["tip_amount"]

        return DatasetSplit(
            x_train_inner,
            x_val_inner,
            x_test_inner,
            y_train_inner,
            y_val_inner,
            y_test_inner,
        )


class MLflowHelper:
    """A helper class to set up and interact with MLflow.

    This class provides utility functions to work with MLflow, including
    resetting the MLflow database, setting up a new MLflow experiment,
    retrieving the best model from the last run in the experiment, and
    registering/promoting the best model to production based on its
    performance compared to the current production model.

    Attributes:
        db_path (str): Path to the MLflow database.
        hpo_experiment_name (str): Name of the hyperparameter optimization
            experiment.
        client (MlflowClient): MLflow tracking client.
        experiment (mlflow.entities.Experiment): MLflow experiment.
        flag_reset_mlflow (str): Flag to determine whether to reset MLflow
            on initialization.

    Methods:
        __init__(self, mlflow_db_path: str, hpo_experiment_name: str,
            args: argparse.Namespace) -> None:
            Constructor for the MLflowHelper class. Resets MLflow if
            specified in args.
        reset_mlflow(self) -> None:
            Reset MLflow by removing the existing MLflow database and
            mlruns artifacts folders.
        setup_mlflow(self) -> mlflow.entities.Experiment:
            Set up MLflow tracking with a new database and create a new
            experiment.
        get_best_model_from_last_run(self) -> mlflow.entities.Run:
            Get the best model from the last run in the MLflow experiment.
        get_production_version(self, registered_model_name: str) ->
            Union[mlflow.entities.model_registry.ModelVersion, None]:
            Get the current production model version.
        run_register_model(self, hpo_champion_model: str,
            x_test_reg: pd.DataFrame, y_test_reg: np.ndarray) -> None:
            Register the best model from the last run and promote it to
            production if better than the current production model.
    """

    def __init__(
        self,
        mlflow_db_path: str,
        hpo_experiment_name: str,
        args_init: argparse.Namespace,
    ):
        self.db_path = mlflow_db_path
        self.hpo_experiment_name = hpo_experiment_name
        self.client = MlflowClient()
        self.experiment = mlflow.get_experiment_by_name(self.hpo_experiment_name)
        self.flag_reset_mlflow = (
            args_init.flag_reset_mlflow
        )  # Get flag_reset_mlflow val from args

        # Call reset_mlflow method during object initialization if specified in args
        if self.flag_reset_mlflow == "Y":
            self.reset_mlflow()

        # Call setup_mlflow method during object initialization
        self.setup_mlflow()

    def reset_mlflow(self) -> None:
        """Reset MLflow by removing the existing MLflow database and mlruns artifacts folders."""
        # Remove existing mlflow.db file
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        directory_path = "./mlruns/"
        # Remove existing mlrun folder
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            # Remove the directory and its contents
            shutil.rmtree(directory_path)

    def setup_mlflow(self) -> mlflow.entities.Experiment:
        """Set up MLflow tracking with a new database and create a new experiment.

        Returns:
            mlflow.entities.Experiment: The created MLflow experiment.
        """
        database_uri = f"sqlite:///{self.db_path}"
        mlflow.set_tracking_uri(database_uri)
        self.experiment = mlflow.set_experiment(self.hpo_experiment_name)
        return self.experiment

    def get_best_model_from_last_run(self) -> mlflow.entities.Run:
        """Get the best model from the last run in the MLflow experiment.

        Returns:
            mlflow.entities.Run: The best model run in the MLflow experiment.
        """
        # Get the maximum existing hpo_run_id in the MLflow database
        existing_runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id]
        )
        existing_hpo_run_ids = [
            int(run.data.tags.get("hpo_run_id", 0)) for run in existing_runs
        ]
        max_hpo_run_id = max(existing_hpo_run_ids) if existing_hpo_run_ids else 0

        # Retrieve the best model run based on the maximum hpo_run_id and test set RMSE
        best_run = None
        best_rmse = float("inf")
        for run in existing_runs:
            hpo_run_id = int(run.data.tags.get("hpo_run_id", 0))
            if hpo_run_id == max_hpo_run_id:
                rmse_test = run.data.metrics.get("RMSE_Test")
                if rmse_test is not None and rmse_test < best_rmse:
                    best_rmse = rmse_test
                    best_run = run

        return best_run

    def get_production_version(
        self, registered_model_name: str
    ) -> Union[mlflow.entities.model_registry.ModelVersion, None]:
        """Get the current production model version.

        Args:
            registered_model_name (str): Name of the registered model.

        Returns:
            Union[mlflow.entities.ModelVersion, None]: The current production model version
            or None if not found.
        """
        try:
            model_versions = self.client.get_latest_versions(
                registered_model_name, stages=["Production"]
            )
            if model_versions:
                production_version = model_versions[0].version
                model_version_details = self.client.get_model_version(
                    registered_model_name, production_version
                )
                return model_version_details
            return None
        except mlflow.exceptions.MlflowException:
            return None

    def get_max_hpo_run_id(self) -> int:
        """Get the maximum existing HPO run ID in the MLflow database for the experiment.

        Returns:
            int: The maximum HPO run ID found in the MLflow database for the specified experiment.
        """
        experiment_id = self.experiment.experiment_id

        existing_runs = self.client.search_runs(experiment_ids=[experiment_id])
        existing_hpo_run_ids = [
            int(run.data.tags.get("hpo_run_id", 0)) for run in existing_runs
        ]
        max_hpo_run_id = max(existing_hpo_run_ids) if existing_hpo_run_ids else 0

        return max_hpo_run_id + 1

    def get_max_model_id(self, hpo_run_id: int) -> int:
        """Get the maximum existing model_id within the specified hpo_run_id.

        Args:
            hpo_run_id (int): The HPO run ID for which to find the maximum model_id.

        Returns:
            int: The maximum model_id found within the specified HPO run ID.
        """
        existing_runs = self.client.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"tags.hpo_run_id = '{hpo_run_id}'",
        )
        existing_model_ids = [
            int(run.data.tags.get("model_id", 0)) for run in existing_runs
        ]
        max_model_id = max(existing_model_ids) if existing_model_ids else 0

        return max_model_id + 1

    def run_register_model(
        self, hpo_champion_model: str, x_test_reg: pd.DataFrame, y_test_reg: np.ndarray
    ) -> None:
        """Register the best model from the last run and promote it to production if better
        than the current production model.

        Args:
            hpo_champion_model (str): Name of the registered model.
            X_test (pd.DataFrame): Test data features.
            y_test (np.ndarray): Test data labels.

        Returns:
            None
        """
        # Retrieve the best model from the last run
        best_run = self.get_best_model_from_last_run()

        # Get the best model URI
        best_model_uri = f"runs:/{best_run.info.run_id}/model"

        # Check if there's already a model in production
        production_version_details = self.get_production_version(hpo_champion_model)

        # If there's a production version, load the production model
        if production_version_details:
            production_pipeline = mlflow.sklearn.load_model(
                production_version_details.source
            )

            # Calculate RMSE for the production model
            y_pred_production = production_pipeline.predict(
                x_test_reg.to_dict(orient="records")
            )
            rmse_production = mean_squared_error(
                y_test_reg, y_pred_production, squared=False
            )

            logger.info("RMSE for the production model: %s", rmse_production)

            # Calculate RMSE for the best model from the last run
            best_pipeline = mlflow.sklearn.load_model(best_model_uri)
            y_pred_best = best_pipeline.predict(x_test_reg.to_dict(orient="records"))
            rmse_best = mean_squared_error(y_test_reg, y_pred_best, squared=False)

            logger.info("RMSE for the best model from the last run: %s", rmse_best)

            # Compare RMSE values and promote the best model if needed
            if rmse_production > rmse_best:
                # Promote the best model from the last run to production
                # Archive the existing production model version
                self.client.transition_model_version_stage(
                    name=hpo_champion_model,
                    version=production_version_details.version,
                    stage="Archived",
                )
                mlflow.register_model(model_uri=best_model_uri, name=hpo_champion_model)
                version = (
                    self.client.get_latest_versions(hpo_champion_model, stages=None)[
                        0
                    ].version
                    + 1
                )
                self.client.transition_model_version_stage(
                    hpo_champion_model, version=str(version), stage="Production"
                )
                logger.info(
                    "The best model from the last run put into production "
                    "and the current model in production archived."
                )
            else:
                # The new model is not better than the one in production,
                # do not register or promote it.
                logger.info(
                    "The new model was not better than the model in production."
                )
        else:
            # If no production version found, register the best model from the last run
            mlflow.register_model(model_uri=best_model_uri, name=hpo_champion_model)
            version = (
                self.client.get_latest_versions(hpo_champion_model, stages=None)[
                    0
                ].version
                + 1
            )
            self.client.transition_model_version_stage(
                hpo_champion_model, version=str(version), stage="Production"
            )
            logger.info("The best model from the last run put into production.")


def run_optimization(
    file_names: Dict[str, Dict[str, str]],
    dataset_split_opt: DatasetSplit,
    mlflow_helper_obj_opt: MLflowHelper,
    num_trials: int,
    pipeline_opt: Pipeline,
) -> None:
    """Run the hyperparameter optimization using Optuna.

    Args:
        dataset_split_opt (DatasetSplit): A data class containing the split datasets and labels.
        client (MlflowClient): MLflow tracking client.
        experiment (mlflow.entities.Experiment): MLflow experiment.
        num_trials (int): Number of hyperparameter optimization trials.
        pipeline_opt (Pipeline): Sklearn pipeline for the model.

    Returns:
        None
    """

    logger.info("Loading and splitting the dataset...")

    x_train_opt, x_val_opt, x_test_opt = (
        dataset_split_opt.x_train,
        dataset_split_opt.x_val,
        dataset_split_opt.x_test,
    )
    y_train_opt, y_val_opt, y_test_opt = (
        dataset_split_opt.y_train,
        dataset_split_opt.y_val,
        dataset_split_opt.y_test,
    )

    # Fit the preprocessor on the training data only
    # once to speed up training (instead of fitting the whole pipeline for each train iteration)
    # # Perhaps this can be achieved with caching option (?)
    pipeline_opt.named_steps["preprocessor"].fit(
        x_train_opt.to_dict(orient="records"), y_train_opt
    )

    # # Preprocess the training data
    x_train_preproc = pipeline_opt.named_steps["preprocessor"].transform(
        x_train_opt.to_dict(orient="records")
    )

    # Build HPO Run id
    hpo_run_id = mlflow_helper_obj_opt.get_max_hpo_run_id()

    def objective(trial):
        params = {
            "random_forest__n_estimators": trial.suggest_int("n_estimators", 10, 50, 1),
            "random_forest__max_depth": trial.suggest_int("max_depth", 1, 20, 1),
            "random_forest__min_samples_split": trial.suggest_int(
                "min_samples_split", 2, 10, 1
            ),
            "random_forest__min_samples_leaf": trial.suggest_int(
                "min_samples_leaf", 1, 4, 1
            ),
            "random_forest__random_state": 42,
            "random_forest__n_jobs": -1,
        }

        # Update the hyperparameters of the model in the pipeline
        pipeline_opt.set_params(**params)

        with mlflow.start_run():
            # Customized run name based on model name and parameters
            params_str = "_".join(
                [f"{param}={value}" for param, value in params.items()]
            )
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            run_name = f"{params_str}_{timestamp}"
            mlflow.set_tag("mlflow.runName", run_name)

            mlflow.set_tag("developer", "nfpaiva")

            mlflow.set_tag("hpo_run_id", str(hpo_run_id))

            mlflow.log_params(params)

            # Train the model part of the pipeline with the preprocessed data
            pipeline_opt.named_steps["random_forest"].fit(x_train_preproc, y_train_opt)

            y_pred_val = pipeline_opt.predict(x_val_opt.to_dict(orient="records"))
            rmse_val = mean_squared_error(y_val_opt, y_pred_val, squared=False)

            y_pred_test = pipeline_opt.predict(x_test_opt.to_dict(orient="records"))
            rmse_test = mean_squared_error(y_test_opt, y_pred_test, squared=False)

            mlflow.log_metric("RMSE_Val", rmse_val)
            logger.info("Trial %s: RMSE_Val = %s", trial.number, rmse_val)

            mlflow.log_metric("RMSE_Test", rmse_test)
            logger.info("Trial %s: RMSE_Test = %s", trial.number, rmse_test)

            model_id = mlflow_helper_obj_opt.get_max_model_id(hpo_run_id)

            mlflow.set_tag("model_id", str(model_id))

            mlflow.set_tag(
                "data_info", str({k: v["month"] for k, v in file_names.items()})
            )

            # log pipeline as an mlflow artifact
            mlflow.sklearn.log_model(sk_model=pipeline_opt, artifact_path="model")

        return rmse_test

    study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=num_trials)


if __name__ == "__main__":
    # Get absolute path of this module and its directory path
    FILE_PATH = Path(__file__).resolve()
    BASE_PATH = FILE_PATH.parent

    config_path = os.path.join(BASE_PATH, "config.yaml")

    with open(config_path, "r", encoding="utf-8") as filename:
        config = yaml.safe_load(filename)

    # attributing required constants
    DATA_DIR = BASE_PATH / "data"
    S3_URL = config["constants"]["S3_URL"]
    PREFIX = config["constants"]["PREFIX"]
    HPO_EXPERIMENT_NAME = config["constants"]["HPO_EXPERIMENT_NAME"]
    HPO_BEST_MODEL = config["constants"]["HPO_BEST_MODEL"]
    HPO_CHAMPION_MODEL = config["constants"]["HPO_CHAMPION_MODEL"]

    parser = argparse.ArgumentParser(description="MLOPS Zoomcamp Homework 2")
    parser.add_argument(
        "--train-month",
        type=str,
        required=True,
        help="Train month in format YYYY-MM",
    )
    parser.add_argument(
        "--val-month",
        type=str,
        required=True,
        help="Validation month in format YYYY-MM",
    )
    parser.add_argument(
        "--test-month",
        type=str,
        required=True,
        help="Test month in format YYYY-MM",
    )
    parser.add_argument(
        "--num-trials",
        type=int,
        default=2,
        help="Number of trials for each Run at mlflow",
    )
    parser.add_argument(
        "--flag-reset-mlflow",
        type=str,
        default="N",
        help="Flag to ask mlflow reset - deleting DB and mlruns artifacts folders. "
        "Y=reset, N=no reset",
    )
    args = parser.parse_args()

    FILE_NAMES = {
        "train": {"month": args.train_month},
        "val": {"month": args.val_month},
        "test": {"month": args.test_month},
    }

    # Instantiate the DataHandler object
    data_handler = DataHandler(DATA_DIR, S3_URL)

    # Download the train, validation, and test data and update the dictionary
    # with corresponding file_paths required for splitting datasets function
    for file_type, month_info in FILE_NAMES.items():
        month = month_info["month"]
        file_name = PREFIX + month + ".parquet"
        month_info["file_name"] = file_name
        month_info["link"] = data_handler.download_data(file_name)

    # Load and split the dataset using the same data_handler instance
    dataset_split = data_handler.load_split_dataset(FILE_NAMES)

    preprocessor = Pipeline(
        [("dict_vectorizer", DictVectorizer(sparse=False))], memory="preprocessor_cache"
    )

    pipeline = Pipeline(
        [("preprocessor", preprocessor), ("random_forest", RandomForestRegressor())]
    )

    # Instantiate MLflowHelper object
    db_path = os.path.join(BASE_PATH, "mlflow.db")
    MLflowHelper_obj = MLflowHelper(db_path, HPO_EXPERIMENT_NAME, args)

    logger.info("Running the optimization...")

    run_optimization(
        FILE_NAMES, dataset_split, MLflowHelper_obj, args.num_trials, pipeline
    )

    logger.info("Running the model registry...")

    MLflowHelper_obj.run_register_model(
        HPO_CHAMPION_MODEL, dataset_split.x_test, dataset_split.y_test
    )
