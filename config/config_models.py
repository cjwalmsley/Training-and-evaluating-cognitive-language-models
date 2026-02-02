# config_models.py
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# Define the path to the directory containing this file (i.e., the 'config' directory)
CONFIG_DIR = Path(__file__).parent


# 1. A nested model for dataset parameters
class DatasetConfig(BaseModel):
    dataset_directory: str
    # store the dataset locally for Linux as the mounted gdrive takes a long time to transfer the dataset files
    dataset_directory_linux: str


class FileLocations(BaseModel):
    base_directory_linux: str
    base_directory_mac: str
    base_directory_windows: str
    prompt_data_directory: str
    prompt_inputs_jsonl_filename: str
    responses_data_directory: str
    responses_jsonl_filename: str
    response_declarative_sentence_categories_filename: str
    response_interrogative_sentence_categories_filename: str
    base_prompt_filename: str
    docker_directory_mac: str
    docker_directory_linux: str
    docker_directory_windows: str
    docker_data_directory: str
    docker_runtime_data_directory: str
    pre_training_directory: str
    training_directory: str
    testing_directory: str
    experiments_directory: str
    log_archive_directory: str
    annabell_log_pre_training_filename: str
    annabell_log_pre_training_validation_testing_filename: str
    annabell_log_training_filename: str
    annabell_log_testing_filename: str
    pre_training_filename: str
    training_filename: str
    testing_filename: str
    results_directory: str
    test_answer_summary_filename: str
    test_detailed_results_filename: str
    test_summary_results_filename: str
    prepared_dataset_directory: str
    prepared_dataset_filename: str
    pre_training_validation_testing_filename: str
    categorised_questions_filename: str
    categorised_statements_filename: str
    prepared_dataset_with_commands_filename: str
    prepared_dataset_pre_commands_filename: str
    classify_sentence_prompt_part_1_filename: str
    classify_sentence_prompt_part_2_filename: str
    sentence_patterns_filename: str
    dataset_with_generated_sentences_filename: str
    dataset_with_sentence_categories_filename: str
    test_pre_training_validation_results_dataframe_filename: str
    test_results_dataframe_filename: str


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


class Experiments(BaseModel):
    experiment_name: str
    percentage_of_pretraining_samples: float
    maximum_number_of_words: int
    maximum_word_length: int
    maximum_phrase_length: int
    number_of_training_samples: int
    use_all_available_samples: bool
    cosine_distance_threshold: float
    auto_save_weights: bool


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
    embedding_model: str
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
    experiments: Experiments