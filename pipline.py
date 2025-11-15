from generate_declarative_sentences import generate_declarative_statements
from categorise_sentences import QuestionCategoryAssigner, StatementCategoryAssigner
from dataset_processing import DatasetPreProcessor
from config.global_config import GlobalConfig
import logging

logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class Pipeline:

    def __init__(self):
        self.declarative_sentences_dataset = None
        self.datasetPreProcessor = None
        self.question_assigner = None
        self.statement_assigner = None
        self.dataset_processor = None
        self.question_categoriser = None
        self.statement_categoriser = None

    def run(self):

        self.generate_declarative_sentences()
        self.preprocess_dataset()
        self.assign_categories()
        self.generate_pre_training_data()
        self.save_prepared_dataset()

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