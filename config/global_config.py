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

    def get_docker_data_directory(self, settings) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class MacConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.base_directory_mac

    def get_docker_data_directory(self, settings) -> str:
        return settings.file_locations.docker_data_directory_mac


class LinuxConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.base_directory_linux

    def get_docker_data_directory(self, settings) -> str:
        return settings.file_locations.docker_data_directory_linux


class WindowsConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.base_directory_windows

    def get_docker_data_directory(self, settings) -> str:
        return settings.file_locations.docker_data_directory_windows


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

        return os.path.join(
            self.get_base_directory(), self.settings.dataset.dataset_directory
        )

    def get_docker_data_directory(self) -> str:

        return self.platform_config.get_docker_data_directory(self.settings)

    def docker_pre_training_directory(self) -> str:

        return os.path.join(
            self.get_docker_data_directory(),
            self.settings.file_locations.pre_training_directory,
        )

    def docker_training_directory(self) -> str:

        return os.path.join(
            self.get_docker_data_directory(),
            self.settings.file_locations.training_directory,
        )

    def docker_testing_directory(self) -> str:

        return os.path.join(
            self.get_docker_data_directory(),
            self.settings.file_locations.testing_directory,
        )

    def pre_training_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.settings.experiment_name,
            self.settings.file_locations.pre_training_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def pre_training_filename(self) -> str:

        return self.settings.file_locations.pre_training_filename

    def pre_training_filepath(self) -> str:

        return os.path.join(
            self.pre_training_directory(),
            self.pre_training_filename(),
        )

    def prepared_dataset_directory(self) -> str:

        directory_path = os.path.join(
            self.get_base_directory(),
            self.settings.file_locations.prepared_dataset_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def prepared_dataset_filepath(self) -> str:

        return os.path.join(
            self.prepared_dataset_directory(),
            self.settings.prepared_dataset_filename,
        )

    def training_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.settings.experiment_name,
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
            self.settings.experiment_name,
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

    def prepared_dataset_with_commands_filepath(self) -> str:

        return os.path.join(
            self.prepared_dataset_directory(),
            self.settings.file_locations.prepared_dataset_with_commands_filepath,
        )

    def categorised_statements_filepath(self) -> str:

        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.categorised_statements_filename,
        )

    def pre_training_validation_testing_filename(self) -> str:

        return self.settings.file_locations.pretraining_validation_testing_filename

    def pretraining_validation_testing_filepath(self) -> str:

        return os.path.join(
            self.testing_directory(),
            self.pre_training_validation_testing_filename(),
        )

    def log_archive_directory(self) -> str:

        directory_path = os.path.join(
            self.get_experiments_directory(),
            self.settings.experiment_name,
            self.settings.file_locations.log_archive_directory,
        )
        if not os.path.exists(directory_path):
            os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def prompt_data_directory(self) -> str:

        directory_path = os.path.join(
            self.get_base_directory(),
            self.settings.file_locations.prompt_data_directory,
        )
        # create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def responses_data_directory(self) -> str:

        directory_path = os.path.join(
            self.get_base_directory(),
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

        return self.settings.ollama.default_model

    def experiment_name(self) -> str:

        return self.settings.experiment_name

    def prepared_dataset_filename(self) -> str:

        return self.settings.prepared_dataset_filename

    def percentage_of_pretraining_samples(self) -> float:

        return self.settings.percentage_of_pretraining_samples

    def maximum_number_of_words(self) -> int:

        return self.settings.maximum_number_of_words

    def maximum_word_length(self) -> int:

        return self.settings.maximum_word_length