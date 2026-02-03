# config/config.py
import os
import sys
import platform
from pathlib import Path
import yaml
from pydantic import ValidationError
from .config_models import Settings
import threading

# --- Path Setup ---
# Find the project root by looking for a known file/directory (e.g., '.git' or 'requirements.txt')
# This makes the path resolution robust, regardless of where scripts are run from.
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_YAML_PATH = PROJECT_ROOT / "config" / "global_config.yml"


def load_settings() -> Settings:
    """
    Loads configuration from the YAML file, environment variables, and .env file.
    - YAML is the base.
    - .env overrides YAML.
    - Environment variables override .env.
    """
    # 1. Load the base config from the YAML file
    try:
        with open(CONFIG_YAML_PATH, "r") as f:
            yaml_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"FATAL: Configuration file not found at {CONFIG_YAML_PATH}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML configuration: {e}")
        sys.exit(1)

    # 2. Instantiate the Settings model.
    # Pydantic automatically merges the YAML data, .env file, and environment variables.
    try:
        settings = Settings.model_validate(yaml_config)
        return settings
    except ValidationError as e:
        print("--- CONFIGURATION VALIDATION ERROR ---")
        print(e)
        sys.exit(1)


# python


class SingletonMeta(type):
    _instances = {}
    _lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            with cls._lock:
                if cls not in cls._instances:
                    cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class AbstractPlatformConfig:
    """
    An abstract base class for platform-specific configuration.
    Subclasses should implement methods to return platform-specific paths.
    """

    def get_base_directory(self, settings) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_docker_directory(self, settings) -> str:
        raise NotImplementedError("Subclasses must implement this method.")

    def get_dataset_directory(self, settings) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class MacConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.base_directory_mac

    def get_docker_directory(self, settings) -> str:
        return settings.file_locations.docker_directory_mac

    def get_dataset_directory(self, settings) -> str:
        return os.path.join(
            self.get_base_directory(settings), settings.dataset.dataset_directory
        )


class LinuxConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.base_directory_linux

    def get_docker_directory(self, settings) -> str:
        return settings.file_locations.docker_directory_linux

    def get_dataset_directory(self, settings) -> str:
        return settings.dataset.dataset_directory_linux


class WindowsConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.base_directory_windows

    def get_docker_directory(self, settings) -> str:
        return settings.file_locations.docker_directory_windows

    def get_dataset_directory(self, settings) -> str:
        return os.path.join(
            self.get_base_directory(settings), settings.dataset.dataset_directory
        )


class GlobalConfig(metaclass=SingletonMeta):

    def __init__(self):
        if getattr(self, "_initialized", False):
            return
        self.settings = load_settings()
        self.project_name = self.settings.project_name
        self.platform_config = self.get_platform_config()
        self._initialized = True

    @staticmethod
    def get_platform_config() -> AbstractPlatformConfig:
        """Sets the platform-specific configuration based on the current OS."""
        os_name = platform.system()
        if os_name == "Windows":
            return WindowsConfig()
        elif os_name == "Linux":
            return LinuxConfig()
        elif os_name == "Darwin":  # macOS
            return MacConfig()
        else:
            raise OSError(f"Unsupported operating system: {os_name}")

    def get_base_directory(self) -> str:
        """Returns the appropriate base directory for the current operating system."""
        return self.platform_config.get_base_directory(self.settings)

    def get_experiments_directory(self) -> str:
        return os.path.join(
            self.get_base_directory(),
            self.settings.file_locations.experiments_directory,
        )

    def dataset_directory(self) -> str:

        return self.platform_config.get_dataset_directory(self.settings)

    def get_docker_directory(self) -> str:

        return self.platform_config.get_docker_directory(self.settings)

    def docker_data_directory(self):
        return os.path.join(
            self.get_docker_directory(),
            self.settings.file_locations.docker_data_directory,
        )

    def docker_runtime_pre_training_directory(self) -> str:
        return os.path.join(
            self.settings.file_locations.docker_runtime_data_directory,
            self.settings.file_locations.pre_training_directory,
        )

    def docker_runtime_training_directory(self) -> str:
        return os.path.join(
            self.settings.file_locations.docker_runtime_data_directory,
            self.settings.file_locations.training_directory,
        )

    def docker_runtime_testing_directory(self) -> str:
        return os.path.join(
            self.settings.file_locations.docker_runtime_data_directory,
            self.settings.file_locations.testing_directory,
        )

    def docker_runtime_pre_training_filepath(self) -> str:

        return os.path.join(
            self.settings.file_locations.docker_runtime_data_directory,
            self.settings.file_locations.pre_training_directory,
            self.settings.file_locations.pre_training_filename,
        )

    def docker_runtime_training_filepath(self) -> str:

        return os.path.join(
            self.docker_runtime_training_directory(),
            self.settings.file_locations.training_filename,
        )

    def docker_runtime_testing_filepath(self) -> str:

        return os.path.join(
            self.docker_runtime_testing_directory(),
            self.settings.file_locations.testing_filename,
        )

    def docker_runtime_pre_training_log_filepath(self) -> str:

        return os.path.join(
            self.docker_runtime_pre_training_directory(),
            self.settings.file_locations.annabell_log_pre_training_filename,
        )

    def docker_runtime_training_log_filepath(self) -> str:

        return os.path.join(
            self.docker_runtime_training_directory(),
            self.settings.file_locations.annabell_log_training_filename,
        )

    def docker_runtime_testing_log_filepath(self) -> str:

        return os.path.join(
            self.docker_runtime_testing_directory(),
            self.settings.file_locations.annabell_log_testing_filename,
        )

    def docker_runtime_pre_training_validation_testing_log_filepath(self) -> str:

        return os.path.join(
            self.docker_runtime_testing_directory(),
            self.settings.file_locations.annabell_log_pre_training_validation_testing_filename,
        )

    def docker_pre_training_directory(self) -> str:

        return os.path.join(
            self.docker_data_directory(),
            self.settings.file_locations.pre_training_directory,
        )

    def docker_training_directory(self) -> str:

        return os.path.join(
            self.docker_data_directory(),
            self.settings.file_locations.training_directory,
        )

    def docker_testing_directory(self) -> str:

        return os.path.join(
            self.docker_data_directory(),
            self.settings.file_locations.testing_directory,
        )

    def docker_pre_training_log_filepath(self) -> str:

        return os.path.join(
            self.docker_pre_training_directory(),
            self.settings.file_locations.annabell_log_pre_training_filename,
        )

    def docker_training_log_filepath(self) -> str:

        return os.path.join(
            self.docker_training_directory(),
            self.settings.file_locations.annabell_log_training_filename,
        )

    def docker_testing_log_filepath(self) -> str:

        return os.path.join(
            self.docker_testing_directory(),
            self.settings.file_locations.annabell_log_testing_filename,
        )

    def docker_pretraining_validation_testing_log_filepath(self) -> str:

        return os.path.join(
            self.docker_testing_directory(),
            self.settings.file_locations.annabell_log_pre_training_validation_testing_filename,
        )

    def pre_training_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.settings.experiments.experiment_name,
            self.settings.file_locations.pre_training_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def pre_training_filename(self) -> str:

        return self.settings.file_locations.pre_training_filename

    def pre_training_weights_filename(self):
        return self.pre_training_filename().replace(".txt", ".dat")

    def pre_training_weights_filepath(self) -> str:

        return os.path.join(
            self.pre_training_directory(),
            self.pre_training_weights_filename(),
        )

    def training_weights_filename(self):
        return self.training_filename().replace(".txt", ".dat")

    def training_weights_filepath(self) -> str:

        return os.path.join(
            self.training_directory(),
            self.training_weights_filename(),
        )

    def docker_pre_training_weights_filepath(self) -> str:
        return os.path.join(
            self.docker_pre_training_directory(),
            self.pre_training_weights_filename(),
        )

    def docker_runtime_pre_training_weights_filepath(self) -> str:

        return os.path.join(
            self.settings.file_locations.docker_runtime_data_directory,
            self.settings.file_locations.pre_training_directory,
            self.pre_training_weights_filename(),
        )

    def docker_runtime_pre_training_validation_testing_filepath(self) -> str:

        return os.path.join(
            self.settings.file_locations.docker_runtime_data_directory,
            self.settings.file_locations.testing_directory,
            self.settings.file_locations.pre_training_validation_testing_filename,
        )

    def docker_runtime_training_weights_filepath(self) -> str:

        return os.path.join(
            self.settings.file_locations.docker_runtime_data_directory,
            self.settings.file_locations.training_directory,
            self.settings.file_locations.training_filename.replace(".txt", ".dat"),
        )

    def docker_pre_training_validation_testing_filepath(self):
        return os.path.join(
            self.docker_testing_directory(),
            self.settings.file_locations.pre_training_validation_testing_filename,
        )

    def docker_testing_filepath(self):
        return os.path.join(
            self.docker_testing_directory(),
            self.settings.file_locations.testing_filename,
        )

    def pre_training_filepath(self) -> str:

        return os.path.join(
            self.pre_training_directory(),
            self.pre_training_filename(),
        )

    def annabell_log_pretraining_validation_testing_filepath(self) -> str:

        return os.path.join(
            self.log_archive_directory(),
            self.settings.file_locations.testing_directory,
            self.settings.file_locations.annabell_log_pretraining_validation_testing_filename,
        )

    def prepared_dataset_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.experiment_name(),
            self.settings.file_locations.prepared_dataset_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def prepared_dataset_filepath(self) -> str:

        return os.path.join(
            self.prepared_dataset_directory(),
            self.prepared_dataset_filename(),
        )

    def prepared_dataset_with_commands_filepath(self) -> str:

        return os.path.join(
            self.prepared_dataset_directory(),
            self.prepared_dataset_with_commands_filename(),
        )

    def prepared_dataset_filepath_exists(self):
        return os.path.exists(self.prepared_dataset_filepath())

    def prepared_dataset_with_commands_filepath_exists(self):
        return os.path.exists(self.prepared_dataset_with_commands_filepath())

    def training_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.experiment_name(),
            self.settings.file_locations.training_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def training_filename(self) -> str:

        return self.settings.file_locations.training_filename

    def training_filepath(self) -> str:

        return os.path.join(
            self.training_directory(),
            self.training_filename(),
        )

    def testing_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.experiment_name(),
            self.settings.file_locations.testing_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def testing_filename(self) -> str:

        return self.settings.file_locations.testing_filename

    def testing_filepath(self) -> str:

        return os.path.join(
            self.testing_directory(),
            self.testing_filename(),
        )

    def categorised_questions_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.categorised_questions_filename,
        )

    def categorised_statements_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.categorised_statements_filename,
        )

    def pre_training_validation_testing_filename(self) -> str:

        return self.settings.file_locations.pre_training_validation_testing_filename

    def test_validation_results_dataframe_filepath(self):
        return os.path.join(
            self.testing_directory(),
            self.settings.file_locations.test_results_dataframe_filename,
        )

    def test_pre_training_validation_results_dataframe_filepath(self):
        return os.path.join(
            self.testing_directory(),
            self.settings.file_locations.test_pre_training_validation_results_dataframe_filename,
        )

    def pre_training_validation_testing_filepath(self) -> str:

        return os.path.join(
            self.testing_directory(),
            self.pre_training_validation_testing_filename(),
        )

    def pre_training_validation_testing_log_filepath(self) -> str:
        return os.path.join(
            self.log_archive_directory(),
            self.settings.file_locations.annabell_log_pre_training_validation_testing_filename,
        )

    def log_archive_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.experiment_name(),
            self.settings.file_locations.log_archive_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def pre_training_log_filename(self) -> str:
        return self.settings.file_locations.annabell_log_pre_training_filename

    def pre_training_log_filepath(self) -> str:

        return os.path.join(
            self.log_archive_directory(),
            self.pre_training_log_filename(),
        )

    def training_log_filepath(self) -> str:

        return os.path.join(
            self.log_archive_directory(),
            self.settings.file_locations.annabell_log_training_filename,
        )

    def testing_log_filepath(self) -> str:

        return os.path.join(
            self.log_archive_directory(),
            self.settings.file_locations.annabell_log_testing_filename,
        )

    def pretraining_validation_testing_log_filepath(self) -> str:

        return os.path.join(
            self.log_archive_directory(),
            self.settings.file_locations.annabell_log_pre_training_validation_testing_filename,
        )

    def prompt_data_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.settings.file_locations.prompt_data_directory,
        )
        # create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def responses_data_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.settings.file_locations.responses_data_directory,
        )
        # create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def responses_jsonl_filepath(self) -> str:

        return os.path.join(
            self.responses_data_directory(),
            self.settings.file_locations.responses_jsonl_filename,
        )

    def base_prompt_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.base_prompt_filename,
        )

    def prompt_inputs_jsonl_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.prompt_inputs_jsonl_filename,
        )

    def classify_sentence_prompt_part_1_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.classify_sentence_prompt_part_1_filename,
        )

    def classify_sentence_prompt_part_2_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.classify_sentence_prompt_part_2_filename,
        )

    def sentence_patterns_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.sentence_patterns_filename,
        )

    def response_declarative_sentence_categories_filepath(self):
        return os.path.join(
            self.responses_data_directory(),
            self.settings.file_locations.response_declarative_sentence_categories_filename,
        )

    def response_interrogative_sentence_categories_filepath(self):
        return os.path.join(
            self.responses_data_directory(),
            self.settings.file_locations.response_interrogative_sentence_categories_filename,
        )

    def dataset_with_generated_sentences_filepath(self) -> str:

        return os.path.join(
            self.prepared_dataset_directory(),
            self.settings.file_locations.dataset_with_generated_sentences_filename,
        )

    def dataset_with_sentence_categories_filepath(self) -> str:

        return os.path.join(
            self.prepared_dataset_directory(),
            self.settings.file_locations.dataset_with_sentence_categories_filename,
        )

    def ollama_model(self) -> str:

        return self.settings.ollama.model

    def ollama_stream(self) -> bool:

        return self.settings.ollama.stream

    def ollama_think(self) -> bool:

        return self.settings.ollama.think

    def wandb_api_key(self) -> str:

        return self.settings.WANDB_API_KEY

    def ollama_options_dict(self) -> dict:

        return self.settings.ollama.options.model_dump()

    def ollama_models(self) -> list[str]:

        return self.settings.ollama.models

    def ollama_default_model(self) -> str:

        return self.settings.ollama.model

    def experiment_name(self) -> str:

        return self.settings.experiments.experiment_name

    def prepared_dataset_filename(self) -> str:

        return self.settings.file_locations.prepared_dataset_filename

    def prepared_dataset_with_commands_filename(self):
        return self.settings.file_locations.prepared_dataset_with_commands_filename

    def prepared_dataset_pre_commands_filename(self):
        return self.settings.file_locations.prepared_dataset_pre_commands_filename

    def percentage_of_pre_training_samples(self) -> float:

        return self.settings.experiments.percentage_of_pretraining_samples

    def maximum_number_of_words(self) -> int:

        return self.settings.experiments.maximum_number_of_words

    def maximum_word_length(self) -> int:

        return self.settings.experiments.maximum_word_length

    def maximum_phrase_length(self) -> int:

        return self.settings.experiments.maximum_phrase_length

    def embedding_model(self):

        return self.settings.ollama.embedding_model

    def number_of_training_samples(self):

        if self.settings.experiments.use_all_available_samples:
            return "all"
        else:
            return self.settings.experiments.number_of_training_samples

    def test_pre_training_validation_results_directory(self):
        directory_name = os.path.join(
            self.pre_training_directory(),
            self.settings.file_locations.results_directory,
        )

        if not os.path.exists(directory_name):
            os.makedirs(directory_name, exist_ok=True)

        return directory_name

    def test_training_results_directory(self):

        directory_name = os.path.join(
            self.training_directory(), self.settings.file_locations.results_directory
        )

        if not os.path.exists(directory_name):
            os.makedirs(directory_name, exist_ok=True)

        return directory_name

    def test_pre_training_validation_answer_summary_filepath(self):
        return os.path.join(
            self.test_pre_training_validation_results_directory(),
            self.settings.file_locations.test_answer_summary_filename,
        )

    def test_pre_training_validation_detailed_results_filepath(self):
        return os.path.join(
            self.test_pre_training_validation_results_directory(),
            self.settings.file_locations.test_detailed_results_filename,
        )

    def test_pre_training_validation_summary_results_filepath(self):
        return os.path.join(
            self.test_pre_training_validation_results_directory(),
            self.settings.file_locations.test_summary_results_filename,
        )

    def test_answer_summary_filepath(self):
        return os.path.join(
            self.test_training_results_directory(),
            self.settings.file_locations.test_answer_summary_filename,
        )

    def test_detailed_results_filepath(self):
        return os.path.join(
            self.test_training_results_directory(),
            self.settings.file_locations.test_detailed_results_filename,
        )

    def test_summary_results_filepath(self):
        return os.path.join(
            self.test_training_results_directory(),
            self.settings.file_locations.test_summary_results_filename,
        )

    def cosine_distance_threshold(self) -> float:
        return self.settings.experiments.cosine_distance_threshold

    @staticmethod
    def project_root():
        return PROJECT_ROOT

    def docker_training_weights_filepath(self):
        return os.path.join(
            self.docker_training_directory(),
            self.training_weights_filename(),
        )

    def auto_save_weights(self) -> bool:
        return self.settings.experiments.auto_save_weights

    def save_weights_every_n_steps(self):
        return self.settings.experiments.save_weights_every_n_steps

    def log_stats_every_n_steps(self):
        return self.settings.experiments.log_stats_every_n_steps