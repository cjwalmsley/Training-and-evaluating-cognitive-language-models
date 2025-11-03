# config/config.py
import os
import sys
import platform
from pathlib import Path
import yaml
from pydantic import ValidationError
from .config_models import Settings

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


class AbstractPlatformConfig:
    """
    An abstract base class for platform-specific configuration.
    Subclasses should implement methods to return platform-specific paths.
    """

    def get_base_directory(self, settings) -> str:
        raise NotImplementedError("Subclasses must implement this method.")


class MacConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.mac_base_directory


class LinuxConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.mac_base_directory


class WindowsConfig(AbstractPlatformConfig):

    def get_base_directory(self, settings) -> str:
        return settings.file_locations.windows_base_directory


class GlobalConfig:
    """
    A convenient, platform-aware wrapper around the main Settings object.
    Provides easy access to platform-specific paths and other configurations.
    """

    def __init__(self):
        self.settings = load_settings()
        self.project_name = self.settings.project_name
        self.platform_config = self.get_platform_config()

    def get_platform_config(self) -> AbstractPlatformConfig:
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

    def dataset_directory(self) -> str:
        """Returns the full, absolute path to the dataset directory."""
        return os.path.join(
            self.get_base_directory(), self.settings.dataset.dataset_directory
        )

    def prompt_data_directory(self) -> str:
        """Returns the full, absolute path to the prompt data directory."""
        directory_path = os.path.join(
            self.get_base_directory(),
            self.settings.file_locations.prompt_data_directory,
        )
        # create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def responses_data_directory(self) -> str:
        """Returns the full, absolute path to the responses data directory."""
        directory_path = os.path.join(
            self.get_base_directory(),
            self.settings.file_locations.responses_data_directory,
        )
        # create the directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        return directory_path

    def responses_jsonl_filepath(self) -> str:
        """Returns the full path to the responses JSONL file."""
        return os.path.join(
            self.responses_data_directory(),
            self.settings.file_locations.responses_jsonl_filename,
        )

    def base_prompt_filepath(self) -> str:
        """Returns the full path to the base prompt file."""
        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.base_prompt_filename,
        )

    def prompt_inputs_jsonl_filepath(self) -> str:
        """Returns the full path to the prompt inputs JSONL file."""
        return os.path.join(
            self.prompt_data_directory(),
            self.settings.file_locations.prompt_inputs_jsonl_filename,
        )

    def ollama_model(self) -> str:
        """Returns the selected Ollama model."""
        return self.settings.ollama.model

    def ollama_stream(self) -> bool:
        """Returns whether Ollama streaming is enabled."""
        return self.settings.ollama.stream

    def ollama_think(self) -> bool:
        """Returns whether Ollama think mode is enabled."""
        return self.settings.ollama.think

    def wandb_api_key(self) -> str:
        """Returns the W&B API key."""
        return self.settings.WANDB_API_KEY

    def ollama_options_dict(self) -> dict:
        """Returns Ollama options as a dictionary."""
        return self.settings.ollama.options.model_dump()

    def ollama_models(self) -> list[str]:
        """Returns the list of available Ollama models."""
        return self.settings.ollama.models

    def ollama_default_model(self) -> str:
        """Returns the default Ollama model."""
        return self.settings.ollama.default_model