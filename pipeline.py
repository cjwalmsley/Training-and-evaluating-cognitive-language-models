from generate_declarative_sentences import generate_declarative_statements
from categorise_sentences import (
    QuestionCategoryAssigner,
    StatementCategoryAssigner,
    QuestionNoCategoryAssigner,
    StatementNoCategoryAssigner,
)
from dataset_processing import DatasetPreProcessor
from training import (
    AnnabellPreTrainingRunner,
    AnnabellPreTrainingTestingRunner,
    AnnabellTrainingRunner,
    AnnabellTestingRunner,
)
from testing import (
    AnnabellTestResultsEvaluator,
    AnnabellPreTrainingTestContext,
    AnnabellTrainingTestContext,
)
from config.global_config import GlobalConfig
import logging
import pandas as pd
import argparse
import os
import subprocess
import time
import json

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class Pipeline:

    def __init__(
        self, use_prepared_dataset_if_available=False, prepared_dataset_filepath=None
    ):
        self.use_prepared_dataset_if_available = use_prepared_dataset_if_available
        self.prepared_dataset_filepath = self.determine_prepared_dataset_filepath(
            prepared_dataset_filepath
        )
        self.declarative_sentences_dataset = None
        self.datasetPreProcessor = None
        self.question_assigner = None
        self.statement_assigner = None
        self.dataset_processor = None
        self.question_categoriser = None
        self.statement_categoriser = None
        self.pre_training_runner = None

    def determine_prepared_dataset_filepath(self, filepath):
        if filepath is not None:
            dataset_filepath = filepath
        elif self.use_prepared_dataset_if_available:
            if global_config.prepared_dataset_pre_commands_filepath_exists():
                dataset_filepath = (
                    global_config.prepared_dataset_pre_commands_filepath()
                )
            else:
                logger.warning(
                    "Prepared dataset file not found. Proceeding without it."
                )
                dataset_filepath = None
        else:
            dataset_filepath = None

        return dataset_filepath

    @staticmethod
    def run_environment_class():
        if global_config.is_hydra():
            return AnnabellHPCApptainerRunEnvironment
        if (
            global_config.local_container_environment() == "apptainer"
            and not global_config.platform_config.is_hydra()
        ):
            return AnnabellLocalApptainerRunEnvironment
        if global_config.local_container_environment() == "docker":
            return AnnabellLocalDockerRunEnvironment
        error_message = "No valid run environment found"
        raise RuntimeError(error_message)

    def run(self):

        logger.info("Starting pipeline...")

        if (
            self.prepared_dataset_filepath is None
            and self.use_prepared_dataset_if_available
        ):
            error_string = "No prepared dataset file provided and use_prepared_dataset_if_available is True, but the default prepared dataset file does not exist. Terminating processing."
            logger.error(error_string)
            raise RuntimeError(error_string)

        if self.prepared_dataset_filepath is None:
            self.generate_declarative_sentences()
            self.preprocess_dataset()
            self.assign_categories()
            self.save_prepared_dataset(
                global_config.prepared_dataset_pre_commands_filename()
            )
        else:
            self.load_prepared_dataset(self.prepared_dataset_filepath)
            self.preprocess_dataset()
        self.generate_pre_training_data()
        self.save_prepared_dataset(
            global_config.prepared_dataset_with_commands_filename()
        )
        self.run_pre_training()
        self.run_pre_training_evaluation_testing()
        self.run_evaluate_pre_training_results()
        self.run_training()
        self.run_testing()
        self.run_evaluate_training_results()
        logger.info("Pipeline completed.")

    def run_training(self):
        logger.info("Starting training...")
        runner = AnnabellTrainingRunner(
            self.datasetPreProcessor, self.run_environment_class()
        )
        runner.run()
        logger.info("Training completed.")

    def run_testing(self):
        logger.info("Starting testing...")
        runner = AnnabellTestingRunner(
            self.datasetPreProcessor, self.run_environment_class()
        )
        runner.run()
        logger.info("Testing completed.")

    def run_pre_training(self):
        logger.info("Starting pre-training...")
        runner = AnnabellPreTrainingRunner(
            self.datasetPreProcessor, self.run_environment_class()
        )

        runner.run()
        logger.info("Pre-training completed.")

    def run_pre_training_evaluation_testing(self):
        logger.info("Starting pre-training testing...")
        runner = AnnabellPreTrainingTestingRunner(
            self.datasetPreProcessor, self.run_environment_class()
        )
        runner.run()
        logger.info("Pre-training testing completed.")

    def run_evaluate_pre_training_results(self):
        logger.info("Starting evaluation of pre-training results...")
        testing_context = AnnabellPreTrainingTestContext(self.datasetPreProcessor)
        evaluator = AnnabellTestResultsEvaluator(testing_context)
        evaluator.run()
        logger.info("Evaluation of pre-training results completed.")

    def run_evaluate_training_results(self):
        logger.info("Starting evaluation of training results...")
        testing_context = AnnabellTrainingTestContext(self.datasetPreProcessor)
        evaluator = AnnabellTestResultsEvaluator(testing_context)
        evaluator.run()
        logger.info("Evaluation of training results completed.")

    def load_prepared_dataset(self, dataset_filepath):
        logger.info(
            f"Loading prepared dataset from {self.prepared_dataset_filepath}..."
        )
        self.declarative_sentences_dataset = pd.read_json(dataset_filepath, lines=True)
        self.datasetPreProcessor = DatasetPreProcessor(
            self.declarative_sentences_dataset.copy()
        )
        logger.info("Prepared dataset loaded successfully.")

    def preprocess_dataset(self):
        logger.info("Starting dataset preprocessing...")
        self.datasetPreProcessor = DatasetPreProcessor(
            self.declarative_sentences_dataset.copy()
        )
        self.datasetPreProcessor.preprocess_data()
        self.declarative_sentences_dataset = self.datasetPreProcessor.dataset
        logger.info("Dataset preprocessing completed.")

    def generate_declarative_sentences(self):
        logger.info("Starting generation of declarative sentences...")
        self.declarative_sentences_dataset = generate_declarative_statements(
            global_config.number_of_training_samples(),
            global_config.ollama_default_model(),
        )
        logger.info("Generation of declarative sentences completed.")

    def generate_pre_training_data(self):
        logger.info("Starting generation of pre-training data...")

        if global_config.categorise_samples():

            self.datasetPreProcessor.select_pretraining_data(
                global_config.percentage_of_pre_training_samples()
            )
        else:
            self.datasetPreProcessor.select_pretraining_data_no_categorisation(
                global_config.percentage_of_pre_training_samples()
            )

        self.datasetPreProcessor.create_commands_for_pretraining()
        self.declarative_sentences_dataset = self.datasetPreProcessor.dataset
        logger.info("Generation of pre-training data completed.")

    def assign_categories(self):
        self.categorise_questions()
        self.categorise_declarative_sentences()

    def categorise_questions(self):

        if global_config.categorise_samples():
            category_assigner_class = QuestionCategoryAssigner
        else:
            category_assigner_class = QuestionNoCategoryAssigner

        logger.info("Starting categorisation of questions...")

        self.question_assigner = category_assigner_class(
            self.declarative_sentences_dataset
        )
        self.question_assigner.generate_statement_categories(
            global_config.ollama_default_model()
        )
        logger.info("Categorisation of questions completed.")

    def categorise_declarative_sentences(self):

        if global_config.categorise_samples():
            category_assigner_class = StatementCategoryAssigner
        else:
            category_assigner_class = StatementNoCategoryAssigner

        logger.info("Starting categorisation of statements...")
        statement_assigner = category_assigner_class(self.declarative_sentences_dataset)
        statement_assigner.generate_statement_categories(
            global_config.ollama_default_model()
        )
        logger.info("Categorisation of statements completed.")

    def save_prepared_dataset(self, filename):

        filepath = os.path.join(global_config.prepared_dataset_directory(), filename)

        self.declarative_sentences_dataset.to_json(
            filepath,
            orient="records",
            lines=True,
        )
        logger.info(f"dataset saved to file: {filepath}")


class AnnabellAbstractRunEnvironment:

    def __init__(self, runner):
        self.runner = runner

    def setup(self):
        self.start_container_environment()

    def teardown(self):
        pass

    def setup_apptainer_directories(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def start_container_environment(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def stop_container_environment(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def copy_annabell_weights(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def move_annabell_logfile(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def pre_training_weights_filepath(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def pre_training_directory(self):
        raise NotImplementedError("Subclasses should implement this method.")


class AnnabellLocalRunEnvironment(AnnabellAbstractRunEnvironment):

    def write_annabell_files(self):
        self.runner.write_annabell_files_to_gdrive()

    def copy_files_to_container_directory(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def run_processing(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def copy_annabell_weights(self):
        self.runner.copy_annabell_weights_to_gdrive()

    def move_annabell_logfile(self):
        self.runner.move_annabell_logfile_to_gdrive()

    def start_container_environment(self):
        raise NotImplementedError("Subclasses should implement this method.")


class AnnabellLocalDockerRunEnvironment(AnnabellLocalRunEnvironment):
    def copy_files_to_container_directory(self):
        self.runner.copy_files_to_docker_directory()

    def run_processing(self):
        self.runner.run_processing_docker()

    def pre_training_weights_filepath(self):
        return global_config.docker_pre_training_weights_filepath()

    @staticmethod
    def training_weights_filepath():
        return global_config.docker_training_weights_filepath()

    @staticmethod
    def testing_log_filepath():
        return global_config.docker_testing_log_filepath()

    @staticmethod
    def training_log_filepath():
        return global_config.docker_training_log_filepath()

    @staticmethod
    def pre_training_validation_testing_log_filepath():
        return global_config.docker_pretraining_validation_testing_log_filepath()

    def pre_training_directory(self):
        return global_config.docker_pre_training_directory()

    def start_container_environment(self):
        self.start_docker_environment()

    def start_docker_environment(self):

        logger.info("Starting Docker container environment...")
        if self.is_docker_running():
            logger.info("Docker is already running.")
        else:
            logger.info("Docker is not running. Attempting to start Docker Desktop...")
            self.start_docker_desktop()

    @staticmethod
    def is_docker_running():
        try:
            subprocess.run(
                ["docker", "info"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def start_docker_desktop(self):
        try:
            subprocess.run(["open", "-a", "Docker"], check=True)
            logger.info("Waiting for Docker to start...")
            while not self.is_docker_running():
                time.sleep(5)
                logger.info("Waiting for Docker daemon...")
            logger.info("Docker started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start Docker Desktop: {e}")
            raise


class AnnabellLocalApptainerRunEnvironment(AnnabellLocalRunEnvironment):

    def setup(self):
        super().setup()
        self.runner.setup_apptainer_directories()

    @staticmethod
    def is_linux_vm_running():
        instance_name = "apptainer"
        result = subprocess.run(
            ["limactl", "list", "--json"],
            check=True,
            capture_output=True,
            text=True,
        )

        result_name = json.loads(result.stdout).get("name")
        result_status = json.loads(result.stdout).get("status")

        if result_name == instance_name and result_status == "Running":
            print(f"Apptainer environment ('{instance_name}') is already running.")
            return True
        else:
            print(f"Environment is '{result_status}'. Starting '{instance_name}'...")
            return False

    @staticmethod
    def run_start_linux_vm():
        subprocess.run(["limactl", "start", "apptainer"], check=True)

    def start_container_environment(self):
        if global_config.platform_config.is_mac():
            self.start_linux_vm()

    def start_linux_vm(self):
        try:
            self.run_start_linux_vm()
            while not self.is_linux_vm_running():
                time.sleep(5)
                logger.info("Waiting for Linux VM to start...")
            logger.info("Linux VM started successfully.")
        except Exception as e:
            logger.critical(f"Failed to start Linux VM: {e}")
            raise

    def copy_files_to_container_directory(self):
        self.runner.copy_files_to_apptainer_directory()

    def run_processing(self):
        self.runner.run_processing_apptainer()

    @staticmethod
    def pre_training_validation_testing_log_filepath():
        return global_config.apptainer_pretraining_validation_testing_log_filepath()

    @staticmethod
    def testing_log_filepath():
        return global_config.apptainer_testing_log_filepath()

    def training_weights_filepath(self):
        return self.runner.apptainer_training_weights_filepath()

    def pre_training_weights_filepath(self):
        return self.runner.apptainer_pre_training_weights_filepath()

    def pre_training_directory(self):
        return global_config.apptainer_pre_training_directory()

    @staticmethod
    def training_log_filepath():
        return global_config.apptainer_training_log_filepath()


class AnnabellHPCRunEnvironment(AnnabellAbstractRunEnvironment):
    def write_annabell_files(self):
        self.runner.write_annabell_files_to_outputs_directory()


class AnnabellHPCApptainerRunEnvironment(AnnabellHPCRunEnvironment):
    def copy_files_to_container_directory(self):
        self.runner.copy_files_to_apptainer_directory()

    def run_processing(self):
        self.runner.run_processing_apptainer()

    def training_weights_filepath(self):
        return self.runner.training_weights_filepath_apptainer()

    @staticmethod
    def testing_log_filepath():
        return global_config.apptainer_testing_log_filepath()

    pass


def main():
    """
    Main entry point for the pipeline script.
    Accepts an optional --prepared-dataset argument to load a pre-prepared dataset.
    """
    parser = argparse.ArgumentParser(description="Run the Annabell training pipeline")
    parser.add_argument(
        "--prepared_dataset_filepath",
        type=str,
        default=None,
        help="Path to a prepared dataset file (JSONL format). If provided, skips data generation and preprocessing steps.",
    )
    parser.add_argument(
        "--use_prepared_dataset_if_available",
        type=bool,
        default=False,
        help="Use the default prepared dataset if available.",
    )

    args = parser.parse_args()

    pipeline = Pipeline(
        use_prepared_dataset_if_available=args.use_prepared_dataset_if_available,
        prepared_dataset_filepath=args.prepared_dataset_filepath,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
