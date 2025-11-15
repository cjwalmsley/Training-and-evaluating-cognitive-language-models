from generate_declarative_sentences import generate_declarative_statements
from categorise_sentences import QuestionCategoryAssigner, StatementCategoryAssigner
from dataset_processing import DatasetPreProcessor
from training import AnnabellPreTrainingRunner
from testing import AnnabellTestResultsEvaluator, AnnabellPreTrainingTestContext
from config.global_config import GlobalConfig
import logging
import pandas as pd
import argparse

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class Pipeline:

    def __init__(self, prepared_dataset_filepath=None):
        self.prepared_dataset_filepath = prepared_dataset_filepath
        self.declarative_sentences_dataset = None
        self.datasetPreProcessor = None
        self.question_assigner = None
        self.statement_assigner = None
        self.dataset_processor = None
        self.question_categoriser = None
        self.statement_categoriser = None
        self.pre_training_runner = None

    def run(self):

        logger.info("Starting pipeline...")
        if self.prepared_dataset_filepath is None:
            self.generate_declarative_sentences()
            self.preprocess_dataset()
            self.assign_categories()
            self.generate_pre_training_data()
            self.save_prepared_dataset()
        else:
            self.load_prepared_dataset()

        self.run_pre_training()
        self.run_evaluate_pre_training_results()
        logger.info("Pipeline completed.")

    def run_pre_training(self):
        logger.info("Starting pre-training...")
        runner = AnnabellPreTrainingRunner(self.datasetPreProcessor)
        runner.run()
        logger.info("Pre-training completed.")

    def run_evaluate_pre_training_results(self):
        logger.info("Starting evaluation of pre-training results...")
        testing_context = AnnabellPreTrainingTestContext(self.datasetPreProcessor)
        evaluator = AnnabellTestResultsEvaluator(testing_context)
        evaluator.run()
        logger.info("Evaluation of pre-training results completed.")

    def load_prepared_dataset(self):
        logger.info(
            f"Loading prepared dataset from {self.prepared_dataset_filepath}..."
        )
        self.declarative_sentences_dataset = pd.read_json(
            self.prepared_dataset_filepath, lines=True
        )
        self.datasetPreProcessor = DatasetPreProcessor(
            self.declarative_sentences_dataset
        )
        logger.info("Prepared dataset loaded successfully.")

    def preprocess_dataset(self):
        logger.info("Starting dataset preprocessing...")
        self.datasetPreProcessor = DatasetPreProcessor(
            self.declarative_sentences_dataset
        )
        self.datasetPreProcessor.preprocess_data()
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
        self.datasetPreProcessor.select_pretraining_data(
            global_config.percentage_of_pre_training_samples()
        )
        self.datasetPreProcessor.create_commands_for_pretraining()
        logger.info("Generation of pre-training data completed.")

    def assign_categories(self):
        self.categorise_questions()
        self.categorise_declarative_sentences()

    def categorise_questions(self):

        logger.info("Starting categorisation of questions...")
        self.question_assigner = QuestionCategoryAssigner(
            self.declarative_sentences_dataset
        )
        self.question_assigner.generate_statement_categories(
            global_config.ollama_default_model()
        )
        logger.info("Categorisation of questions completed.")

    def categorise_declarative_sentences(self):

        logger.info("Starting categorisation of statements...")
        statement_assigner = StatementCategoryAssigner(
            self.declarative_sentences_dataset
        )
        statement_assigner.generate_statement_categories(
            global_config.ollama_default_model()
        )
        logger.info("Categorisation of statements completed.")

    def save_prepared_dataset(self):

        self.declarative_sentences_dataset.to_json(
            global_config.prepared_dataset_with_commands_filepath(),
            orient="records",
            lines=True,
        )
        logger.info(
            f"dataset saved to file: {global_config.prepared_dataset_with_commands_filepath()}"
        )


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

    args = parser.parse_args()

    pipeline = Pipeline(prepared_dataset_filepath=args.prepared_dataset_filepath)
    pipeline.run()


if __name__ == "__main__":
    main()