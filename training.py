import subprocess
import shlex
from config.global_config import GlobalConfig
import logging
import shutil
import os


logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class AbstractAnnabellRunner:

    def __init__(self, annabell_commands_filename, dataset_processor):
        self.commands_filename = annabell_commands_filename
        self.dataset_processor = dataset_processor

    def run(
        self,
    ):
        self.setup()
        self.run_processing()
        self.teardown()

    def setup(self):
        self.write_annabell_files_to_gdrive()
        self.copy_files_to_docker_directory()

    def write_annabell_files_to_gdrive(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def teardown(self):
        pass

    def copy_files_to_docker_directory(self):

        raise NotImplementedError("Subclasses should implement this method.")

    def run_script(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def docker_command(self):

        command = (
            f"docker compose run --rm --entrypoint ./{self.run_script()} app "
            f"{self.commands_filename} "
            f"{global_config.docker_data_directory()}/{global_config.pre_training_filename()} "
            f"{global_config.docker_data_directory()}/{global_config.pre_training_weights_filename()}"
        )
        return command

    def copy_file(self, source_path, destination_path):

        try:
            shutil.copy(source_path, destination_path)
            logger.info("copied: " + source_path + " to: " + destination_path)
        except FileNotFoundError:
            logger.critical(f"Error: Source file not found at {source_path}")
        except Exception as e:
            logger.critical(f"An error occurred: {e}")

    def move_file(self, source_path, destination_path):

        try:
            shutil.move(source_path, destination_path)
            logger.info("moved: " + source_path + " to: " + destination_path)
        except FileNotFoundError:
            logger.critical(f"Error: Source file not found at {source_path}")
        except Exception as e:
            logger.critical(f"An error occurred: {e}")

    def run_processing(self):
        # Use shlex.split to handle arguments safely
        command_args = shlex.split(self.docker_command())

        try:
            # Execute the command
            # Using check=True will raise an exception if the command returns a non-zero exit code.
            result = subprocess.run(
                command_args, check=True, text=True, capture_output=True
            )
            logger.info("STDOUT:", result.stdout)
            logger.info("STDERR:", result.stderr)
            logger.info("\nDocker command executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.critical(f"Docker command failed with exit code {e.returncode}")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
        except FileNotFoundError:
            logger.critical(
                "Error: 'docker' command not found. Make sure Docker is installed and in your system's PATH."
            )
        except Exception as e:
            logger.critical(f"An unexpected error occurred: {e}")


class AnnabellPreTrainingRunner(AbstractAnnabellRunner):

    def setup(self):
        super().setup()

    def teardown(self):
        self.copy_annabell_weights_to_gdrive()
        self.move_annabell_logfile_to_gdrive()
        super().teardown()

    def run_script(self):
        return "pre_train_annabell_squad_nyc.sh"

    def docker_command(self):

        command = (
            f"docker compose run --rm --entrypoint ./{self.run_script()} app "
            f"{global_config.docker_runtime_pre_training_log_filepath()} "
            f"{global_config.docker_runtime_pre_training_filepath()} "
            f"{global_config.docker_runtime_pre_training_weights_filepath()}"
        )
        return command

    def write_annabell_files_to_gdrive(self):

        self.dataset_processor.write_pretraining_file(
            global_config.pre_training_filepath()
        )

    def copy_files_to_docker_directory(self):
        self.copy_file(
            global_config.pre_training_filepath(),
            global_config.docker_pre_training_directory(),
        )

    def copy_annabell_weights_to_gdrive(self):
        # copy the pre-trained weights to the pre-training directory
        source_path = os.path.join(
            global_config.docker_pre_training_directory(),
            global_config.pre_training_weights_filename(),
        )
        destination_path = os.path.join(
            global_config.pre_training_directory(),
            global_config.pre_training_weights_filename(),
        )

        self.copy_file(source_path, destination_path)

    def move_annabell_logfile_to_gdrive(self):
        source_path = os.path.join(
            global_config.docker_pre_training_directory(),
            global_config.pre_training_log_filename(),
        )
        destination_path = global_config.pre_training_log_filepath()

        self.move_file(source_path, destination_path)


class AnnabellTestingRunner(AbstractAnnabellRunner):

    def run_script(self):
        return "test_annabell_squad_nyc.sh"

    def run_processing(self):
        pass

    def copy_files_to_docker_directory(self):
        self.copy_file(
            global_config.pre_training_validation_testing_filepath(),
            global_config.docker_pretraining_validation_testing_filepath(),
        )

    def move_annabell_logfile_to_gdrive(self):
        source_path = os.path.join(
            global_config.docker_pretraining_validation_testing_log_filepath(),
        )
        destination_path = global_config.pretraining_validation_testing_log_filepath()

        self.move_file(source_path, destination_path)

    def docker_command(self):

        command = (
            f"docker compose run --rm --entrypoint ./{self.run_script()} app"
            f"{global_config.docker_runtime_pre_training_log_filepath()}"
            f"{global_config.docker_runtime_pre_training_weights_filepath()}"
            f"{global_config.docker_runtime_pre_training_validation_testing_filepath()}"
        )

        return command


class AnnabellTrainingRunner(AbstractAnnabellRunner):

    def run_script(self):
        return "train_annabell_squad_nyc.sh"

    def run_processing(self):
        pass