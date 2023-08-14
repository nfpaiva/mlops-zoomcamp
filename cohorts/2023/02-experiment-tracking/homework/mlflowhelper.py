"""
mlflow_helper.py: A module for managing MLflow experiments and model lifecycle.

This module provides classes and functions to manage MLflow experiments, track model runs,
and handle model lifecycle transitions. It includes functionality to reset the MLflow tracking
database, set up experiments, find the best model from a series of runs, and manage model versions
in different stages such as production and archived.

Classes:
    - NoExperimentFound: Exception raised when no experiment is found.
    - NoRunsFound: Exception raised when no runs are found.
    - NoModelsFound: Exception raised when no models are found.
    - MlFlowContext: Manages the setup of MLflow environment and experiment context.
    - MlFlowExperimentRegistry: Manages experiment-related queries and operations.
    - MlFlowModelManager: Manages model-related queries, calculations, and transitions.

Functions:
    - compare_models: Compares production and best model RMSE values.
    - handle_challenger_wins: Handles model version transitions when the challenger wins.

Usage:
    Import this module to manage MLflow experiments, model runs, and lifecycle transitions.
    Create an instance of MlFlowContext, MlFlowExperimentRegistry, and MlFlowModelManager
    to perform various MLflow-related tasks.

"""

import os
import shutil
import argparse
import logging


from typing import Union, Tuple

import pandas as pd
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient  # type: ignore
from mlflow.entities import Run, model_registry  # type: ignore
from mlflow.entities.experiment import Experiment  # type: ignore

from sklearn.metrics import mean_squared_error  # type: ignore

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NoExperimentFound(Exception):
    """Exception raised when no experiment is found."""


class NoRunsFound(Exception):
    """Exception raised when no runs are found."""


class NoModelsFound(Exception):
    """Exception raised when no models are found."""


class MlFlowContext:
    """
    Manages the setup of the MLflow environment and experiment context.

    This class handles the initialization of the MLflow tracking database, experiment context,
    and related configurations. It provides methods to reset the tracking database, set up the
    MLflow environment, and retrieve the experiment associated with the provided experiment name.

    Args:
        mlflow_db_path (str): Path to the SQLite database file for MLflow tracking.
        hpo_experiment_name (str): Name of the hyperparameter optimization (HPO) experiment.
        args_init (argparse.Namespace): Parsed command-line arguments.

    Attributes:
        db_path (str): Path to the MLflow tracking database.
        hpo_experiment_name (str): Name of the HPO experiment.
        flag_reset_mlflow (str): Flag indicating whether to reset the MLflow environment.

    Methods:
        reset_mlflow: Resets the MLflow tracking database and associated directories.
        setup_mlflow: Sets up the MLflow tracking environment and returns the tracking client.
        _get_experiment_or_raise: Retrieves the experiment by name or creates a new one.

    Usage:
        mlflow_context = MlFlowContext(mlflow_db_path, hpo_experiment_name, args_init)
        experiment = mlflow_context.experiment  # Access the experiment context.
    """

    def __init__(
        self,
        mlflow_db_path: str,
        hpo_experiment_name: str,
        args_init: argparse.Namespace,
    ):
        """
        Initialize an MlFlowContext instance.

        This constructor initializes an MlFlowContext instance by setting up the necessary
        attributes and calling relevant setup methods.

        Args:
            mlflow_db_path (str): Path to the SQLite database file for MLflow tracking.
            hpo_experiment_name (str): Name of the hyperparameter optimization (HPO) experiment.
            args_init (argparse.Namespace): Parsed command-line arguments.

        Attributes:
            db_path (str): Path to the MLflow tracking database.
            hpo_experiment_name (str): Name of the HPO experiment.
            flag_reset_mlflow (str): Flag indicating whether to reset the MLflow environment.

        Returns:
            None
        """
        self.db_path = mlflow_db_path
        self.hpo_experiment_name = hpo_experiment_name
        self.flag_reset_mlflow = (
            args_init.flag_reset_mlflow
        )  # Get flag_reset_mlflow val from args

        # Call reset_mlflow method during object initialization if specified in args
        if self.flag_reset_mlflow == "Y":
            self.reset_mlflow()
        # Call setup_mlflow method during object initialization
        self.setup_mlflow()
        self._get_experiment_or_raise()

    def reset_mlflow(self) -> None:
        """
        Reset the MLflow environment by removing existing database and run data.

        This method deletes the existing MLflow tracking database file and removes the
        directory containing the MLflow run data, effectively resetting the MLflow environment.

        Args:
            None

        Returns:
            None
        """

        # Remove existing mlflow.db file
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

        directory_path = "./mlruns/"
        # Remove existing mlrun folder
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            # Remove the directory and its contents
            shutil.rmtree(directory_path)

    def setup_mlflow(self) -> Tuple[Experiment, mlflow.tracking.MlflowClient]:
        """
        Set up the MLflow tracking environment and create a client.

        This method configures the MLflow tracking URI to use a SQLite database located
        at the specified path. It creates an instance of the MLflow tracking client to
        interact with the configured tracking environment.

        Args:
            None

        Returns:
            Tuple containing the MLflow Experiment and MlflowClient instances.
        """

        database_uri = f"sqlite:///{self.db_path}"
        mlflow.set_tracking_uri(database_uri)
        self.client_mlflow = MlflowClient(tracking_uri=database_uri)

        return self.client_mlflow

    def _get_experiment_or_raise(self) -> Experiment:
        """
        Retrieve the MLflow Experiment associated with the given experiment name.

        This method checks if the experiment needs to be reset based on the provided flag.
        If reset is required, a new experiment is created with the specified name.
        If reset is not required, it tries to retrieve an existing experiment with the
        given name. If the experiment is not found, a NoExperimentFound exception is raised.

        Args:
            None

        Returns:
            Experiment: The MLflow Experiment instance associated with the experiment name.

        Raises:
            NoExperimentFound: If the experiment with the given name does not exist.
        """

        if self.flag_reset_mlflow == "Y":
            self.experiment = mlflow.set_experiment(self.hpo_experiment_name)
        else:
            self.experiment = mlflow.get_experiment_by_name(self.hpo_experiment_name)
            if self.experiment is None:
                raise NoExperimentFound(
                    f"No experiment found with name {self.hpo_experiment_name}."
                )
        return self.experiment


class MlFlowExperimentRegistry:
    """
    Helper class to interact with MLflow experiments and retrieve information about runs.

    This class provides methods to retrieve information about hyperparameter optimization (HPO)
    runs and models associated with those runs within a specific MLflow experiment.

    Args:
        mlflow_context (MlFlowContext): An instance of the MlFlowContext class providing
            the MLflow tracking configuration and experiment details.

    Attributes:
        experiment (Experiment): The MLflow Experiment instance associated with the experiment name.
        client_mlflow (MlflowClient): The MLflow Tracking client for interacting
        with runs and models.
    """

    def __init__(self, mlflow_context: MlFlowContext):
        """
        Initialize an instance of MlFlowExperimentRegistry with the provided MlFlowContext.

        Args:
            mlflow_context (MlFlowContext): An instance of MlFlowContext containing the experiment
                details and tracking configuration.
        """

        self.experiment = mlflow_context.experiment
        self.client_mlflow = mlflow_context.client_mlflow

    def get_max_hpo_run_id(self) -> int:
        """
        Retrieve the maximum hyperparameter optimization (HPO) run ID within the experiment.

        This method searches for existing HPO runs within the experiment and calculates the maximum
        HPO run ID present. It helps in generating new HPO run IDs for subsequent runs.

        Returns:
            int: The maximum HPO run ID within the experiment incremented by 1.
        """
        experiment_id = self.experiment.experiment_id
        existing_runs = self.client_mlflow.search_runs(experiment_ids=[experiment_id])
        existing_hpo_run_ids = [
            int(run.data.tags.get("hpo_run_id", 0)) for run in existing_runs
        ]
        max_hpo_run_id = max(existing_hpo_run_ids) if existing_hpo_run_ids else 0

        return max_hpo_run_id + 1

    def get_max_model_id(self, hpo_run_id: int) -> int:
        """
        Retrieve the maximum model ID associated with a specific HPO run.

        This method searches for existing runs within the experiment with the specified HPO run ID
        and calculates the maximum model ID associated with those runs. It helps in generating new
        model IDs for models produced by the HPO process.

        Args:
            hpo_run_id (int): The HPO run ID for which to retrieve the maximum model ID.

        Returns:
            int: The maximum model ID associated with the specified HPO run incremented by 1.
        """
        existing_runs = self.client_mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            filter_string=f"tags.hpo_run_id = '{hpo_run_id}'",
        )
        existing_model_ids = [
            int(run.data.tags.get("model_id", 0)) for run in existing_runs
        ]
        max_model_id = max(existing_model_ids) if existing_model_ids else 0

        return max_model_id + 1


class MlFlowModelManager:
    """
    Helper class to manage models, promote models to production, and compare models using MLflow.

    This class provides methods to manage models produced through hyperparameter optimization (HPO)
    runs, promote models to production, and compare different models using RMSE values.

    Args:
        mlflow_context (MlFlowContext): An instance of the MlFlowContext class providing
            the MLflow tracking configuration and experiment details.

    Attributes:
        experiment (Experiment): The MLflow Experiment instance associated with the experiment name.
        client_mlflow (MlflowClient): The MLflow Tracking client for interacting
        with runs and models.
    """

    def __init__(self, mlflow_context: MlFlowContext):
        """
        Initialize an instance of MlFlowModelManager with the provided MlFlowContext.

        Args:
            mlflow_context (MlFlowContext): An instance of MlFlowContext containing the experiment
                details and tracking configuration.
        """
        self.experiment = mlflow_context.experiment
        self.client_mlflow = mlflow_context.client_mlflow

    def get_best_model_from_last_run(self) -> Run:
        """
        Retrieve the best model from the last hyperparameter optimization (HPO) run.

        This method searches for all runs within the experiment and identifies the best model
        produced by the latest HPO run based on the RMSE metric. The best model is determined
        by comparing RMSE values of different runs.

        Returns:
            Run: The Run object representing the best model from the last HPO run.
        """
        # Get the maximum existing hpo_run_id in the MLflow database

        existing_runs = self.client_mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id]
        )
        if not existing_runs:
            raise NoRunsFound(
                f"No runs found in experiment {self.experiment.experiment_id}"
            )

        existing_hpo_run_ids = [
            int(run.data.tags.get("hpo_run_id", 0)) for run in existing_runs
        ]
        max_hpo_run_id = max(existing_hpo_run_ids) if existing_hpo_run_ids else 0

        # Retrieve the best model run based on the maximum hpo_run_id and test set RMSE
        # The first run in the list is the best run by default
        best_run = existing_runs[0]
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
    ) -> Union[model_registry.ModelVersion, None]:
        """
        Retrieve the current production version of a registered model.

        This method queries the MLflow registry to identify the current production version of a
        registered model based on its name. It returns the details of the production model version,
        or None if no production version exists.

        Args:
            registered_model_name (str): The name of the registered model to retrieve
            production version for.

        Returns:
            Optional[model_registry.ModelVersion]: The ModelVersion instance representing the
            production version,
                or None if no production version exists.
        """
        try:
            model_versions = self.client_mlflow.get_latest_versions(
                registered_model_name, stages=["Production"]
            )
            if model_versions:
                production_version = model_versions[0].version
                model_version_details = self.client_mlflow.get_model_version(
                    registered_model_name, production_version
                )
                return model_version_details
            return None
        except mlflow.exceptions.MlflowException:
            return None

    def calc_rmse_champion_challenger_model(
        self,
        hpo_champion_model: str,
        x_test_reg: pd.DataFrame,
        y_test_reg: Union[np.ndarray, pd.Series],
        best_model_uri: str,
        best_run: mlflow.entities.Run,
        production_pipeline: mlflow.pyfunc.PythonModel,
    ) -> Tuple[float, float]:
        """
        Calculate RMSE values for the production champion model and the best challenger model.

        Args:
            hpo_champion_model (str): The name of the production champion model.
            x_test_reg (pd.DataFrame): Test feature data for model evaluation.
            y_test_reg (Union[np.ndarray, pd.Series]): True labels for model evaluation.
            best_model_uri (str): URI of the best challenger model from the last HPO run.
            best_run (mlflow.entities.Run): The Run object representing the
            best challenger model's run.
            production_pipeline (mlflow.pyfunc.PythonModel): The production champion
            model's pipeline.

        Returns:
            Tuple[float, float]: A tuple containing the RMSE values for the production
            champion model and
                the best challenger model.
        """

        y_pred_production = production_pipeline.predict(
            x_test_reg.to_dict(orient="records")
        )
        rmse_production = mean_squared_error(
            y_test_reg, y_pred_production, squared=False
        )

        logger.info("RMSE for the production - champion - model: %s", rmse_production)

        # Calculate RMSE for the best model from the last run
        best_pipeline = mlflow.sklearn.load_model(best_model_uri)
        if not best_pipeline:
            raise NoModelsFound(
                f"No best model found for {hpo_champion_model} in run {best_run.info.run_id}"
            )
        y_pred_best = best_pipeline.predict(x_test_reg.to_dict(orient="records"))
        rmse_best = mean_squared_error(y_test_reg, y_pred_best, squared=False)

        logger.info(
            "RMSE for the best model - challenger - from the last run: %s", rmse_best
        )
        return rmse_production, rmse_best

    def handle_challenger_wins(
        self,
        hpo_champion_model: str,
        best_model_uri: str,
        production_version_details: model_registry.ModelVersion,
    ) -> None:
        """
        Handle the scenario where the best challenger model outperforms the
        production champion model.

        Args:
            hpo_champion_model (str): The name of the production champion model.
            best_model_uri (str): URI of the best challenger model from the last HPO run.
            production_version_details (model_registry.ModelVersion): Details of the current
            production version.

        Returns:
            None
        """

        self.client_mlflow.transition_model_version_stage(
            name=hpo_champion_model,
            version=production_version_details.version,
            stage="Archived",
        )
        mlflow.register_model(model_uri=best_model_uri, name=hpo_champion_model)
        version = (
            self.client_mlflow.get_latest_versions(hpo_champion_model, stages=None)[
                0
            ].version
            + 1
        )
        self.client_mlflow.transition_model_version_stage(
            hpo_champion_model, version=str(version), stage="Production"
        )
        logger.info(
            "The best model from the last run put into production "
            "and the current model in production archived."
        )

    def run_register_model(
        self,
        hpo_champion_model: str,
        x_test_reg: pd.DataFrame,
        y_test_reg: Union[np.ndarray, pd.Series],
    ) -> None:
        """
        Register the best model from the last run or promote it to production.

        Args:
            hpo_champion_model (str): The name of the production champion model.
            x_test_reg (pd.DataFrame): Test feature data for model evaluation.
            y_test_reg (Union[np.ndarray, pd.Series]): True labels for model evaluation.

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

            if not production_pipeline:
                raise NoModelsFound(
                    f"No production model found for {hpo_champion_model}"
                )
            rmse_production, rmse_best = self.calc_rmse_champion_challenger_model(
                hpo_champion_model,
                x_test_reg,
                y_test_reg,
                best_model_uri,
                best_run,
                production_pipeline,
            )
            # Compare RMSE values and promote the best model if needed
            if rmse_production > rmse_best:
                # Promote the best model from the last run to production
                # Archive the existing production model version

                self.handle_challenger_wins(
                    hpo_champion_model, best_model_uri, production_version_details
                )

            else:
                # The new model is not better than the one in production,
                # do not register or promote it.
                logger.info(
                    "The new model was not better than the model in production."
                    "The current model in production remains unchanged."
                )

            # Add a custom tag to the best model URI to indicate the outcome of the comparison
            custom_tag = {
                "comparison_outcome": "better_than_production"
                if rmse_best < rmse_production
                else "not_better_than_production",
                "production_model_uri": str(production_version_details.run_id),
                "test_rmse": str(rmse_best),
            }

            for key, value in custom_tag.items():
                self.client_mlflow.set_tag(best_run.info.run_id, key, value)

        else:
            logger.info(
                "No model exists in production. The best model from the "
                "last run will be put into production."
            )
            mlflow.register_model(model_uri=best_model_uri, name=hpo_champion_model)
            self.client_mlflow.transition_model_version_stage(
                name=hpo_champion_model, version="1", stage="Production"
            )
