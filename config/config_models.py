# config_models.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Define the path to the directory containing this file (i.e., the 'config' directory)
CONFIG_DIR = Path(__file__).parent


# 1. A nested model for dataset parameters
class DatasetConfig(BaseModel):
    dataset_directory: str


class FileLocations(BaseModel):
    base_directory_windows: str
    base_directory_linux: str
    base_directory_mac: str
    prompt_data_directory: str
    prompt_inputs_jsonl_filename: "str"
    responses_data_directory: str
    responses_jsonl_filename: str
    base_prompt_filename: str


class OllamaOptions(BaseModel):
    num_ctx: int = 4096
    repeat_last_n: int = 64
    repeat_penalty: float = 1.5
    temperature: float = 0
    seed: int = 42
    num_predict: int = 100
    top_k: int = 1
    top_p: float = 0.1
    min_p: float = 0.0


# 2. A nested model for model hyperparameters
class OllamaConfig(BaseModel):
    options: OllamaOptions
    models: list[str] = [
        "llama3.2",
        "llama3",
        "deepseek-r1:8b",
        "gemma3:4b",
        "gemma3:1b",
        "gemma3n:e4b",
        "qwen3:4b",
    ]
    model: str
    stream: bool
    think: bool


# 3. The main settings class that inherits from BaseSettings
class Settings(BaseSettings):
    # This tells Pydantic to look for a .env file in the same directory as this script
    model_config = SettingsConfigDict(
        env_file=CONFIG_DIR / ".env", env_file_encoding="utf-8"
    )

    # This setting is *required* but we expect
    # it to come from the .env file, NOT the YAML.
    WANDB_API_KEY: str

    # These settings will be loaded from YAML
    project_name: str
    dataset: DatasetConfig
    ollama: OllamaConfig
    file_locations: FileLocations