import subprocess
import shlex
from config.global_config import GlobalConfig, PROJECT_ROOT
import logging
import shutil
import os


logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class AbstractAnnabellRunner:

    def __init__(self, dataset_processor):
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
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def copy_file(source_path, destination_path):

        try:
            shutil.copy(source_path, destination_path)
            logger.info("copied: " + source_path + " to: " + destination_path)
        except Exception as e:
            logger.critical(f"Copy failed: {source_path} -> {destination_path}")
            logger.critical(f"An error occurred: {e}")
            raise

    @staticmethod
    def move_file(source_path, destination_path):

        try:
            shutil.move(source_path, destination_path)
            logger.info("moved: " + source_path + " to: " + destination_path)
        except Exception as e:
            logger.critical(f"Move failed: {source_path} -> {destination_path}")
            logger.critical(f"An error occurred: {e}")
            raise

    def run_processing(self):
        # Use shlex.split to handle arguments safely
        command_args = shlex.split(self.docker_command())
        docker_directory = os.path.join(PROJECT_ROOT, "docker")

        # Verify docker directory exists
        if not os.path.isdir(docker_directory):
            raise FileNotFoundError(f"Docker directory not found: {docker_directory}")

        # Find the full path to docker executable to avoid PyCharm debugger interference
        docker_path = shutil.which("docker")
        if not docker_path:
            raise FileNotFoundError(
                "'docker' command not found. Make sure Docker is installed and in your system's PATH."
            )

        # Replace 'docker' with full path in command_args
        if command_args[0] == "docker":
            command_args[0] = docker_path

        logger.info(f"Running Docker command from directory: {docker_directory}")
        logger.info(f"Command: {' '.join(command_args)}")

        try:
            # Execute the command
            # Using check=True will raise an exception if the command returns a non-zero exit code.
            # shell=False ensures command_args is treated as a list, not a shell command
            result = subprocess.run(
                command_args,
                check=True,
                text=True,
                capture_output=True,
                cwd=docker_directory,
                shell=False,
            )
            logger.info("STDOUT: %s", result.stdout)
            logger.info("STDERR: %s", result.stderr)
            logger.info("\nDocker command executed successfully.")
        except subprocess.CalledProcessError as e:
            logger.critical(f"Docker command failed with exit code {e.returncode}")
            logger.error("STDOUT: %s", e.stdout)
            logger.error("STDERR: %s", e.stderr)
            raise
        except FileNotFoundError as e:
            logger.critical(
                "Error: 'docker' command not found. Make sure Docker is installed and in your system's PATH."
            )
            logger.critical(f"Details: {e}")
            raise
        except Exception as e:
            logger.critical(f"An unexpected error occurred: {e}")
            raise

    def docker_runtime_weights_filepath(self):
        raise NotImplementedError("Subclasses should implement this method.")


class AnnabellPreTrainingRunner(AbstractAnnabellRunner):

    def setup(self):
        super().setup()
        self.write_annabell_files_to_gdrive()
        self.copy_files_to_docker_directory()

    def teardown(self):
        self.copy_annabell_weights_to_gdrive()
        self.move_annabell_logfile_to_gdrive()
        super().teardown()

    def run_script(self):
        return "pre_train_annabell_squad_nyc.sh"

    def docker_command(self):

        command = (
            f"docker compose run --rm --remove-orphans --entrypoint ./{self.run_script()} app "
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
        source_path = os.path.join(global_config.docker_pre_training_weights_filepath())
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
        return "test_annabell_squad.sh"

    def copy_annabell_weights_to_gdrive(self):
        pass

    def setup(self):
        super().setup()
        self.write_annabell_files_to_gdrive()
        self.copy_files_to_docker_directory()

    def teardown(self):
        self.move_annabell_logfile_to_gdrive()
        super().teardown()

    def docker_command(self):
        command = (
            f"docker compose run --rm --entrypoint ./{self.run_script()} app "
            f"{self.log_filepath()} "
            f"{self.commands_filepath} "
            f"{self.docker_runtime_weights_filepath()}"
        )
        return command

    def copy_files_to_docker_directory(self):
        self.copy_file(
            global_config.testing_filepath(),
            global_config.docker_testing_filepath(),
        )

    def move_annabell_logfile_to_gdrive(self):
        source_path = os.path.join(
            global_config.docker_testing_log_filepath(),
        )
        destination_path = global_config.testing_log_filepath()

        self.move_file(source_path, destination_path)


class AnnabellPreTrainingTestingRunner(AnnabellTestingRunner):

    def docker_command(self):
        command = (
            f"docker compose run --rm --entrypoint ./{self.run_script()} app "
            f"{self.docker_runtime_log_filepath()} "
            f"{self.docker_runtime_weights_filepath()} "
            f"{self.docker_runtime_commands_filepath()}"
        )
        return command

    def docker_runtime_weights_filepath(self):
        return global_config.docker_runtime_pre_training_weights_filepath()

    def docker_runtime_commands_filepath(self):
        return global_config.docker_runtime_pre_training_validation_testing_filepath()

    def docker_runtime_log_filepath(self):
        return (
            global_config.docker_runtime_pre_training_validation_testing_log_filepath()
        )

    def write_annabell_files_to_gdrive(self):
        self.dataset_processor.write_pretraining_testing_file(
            global_config.pre_training_validation_testing_filepath()
        )

    def copy_files_to_docker_directory(self):
        self.copy_file(
            global_config.pre_training_validation_testing_filepath(),
            global_config.docker_pre_training_validation_testing_filepath(),
        )
        self.copy_file(
            global_config.pre_training_weights_filepath(),
            global_config.docker_pre_training_weights_filepath(),
        )

    def move_annabell_logfile_to_gdrive(self):
        source_path = global_config.docker_pretraining_validation_testing_log_filepath()
        destination_path = global_config.pre_training_validation_testing_log_filepath()

        self.move_file(source_path, destination_path)


class AnnabellTrainingRunner(AbstractAnnabellRunner):

    def run_script(self):
        return "train_annabell_squad_nyc.sh"

    def run_processing(self):
        pass