import subprocess
import shlex
from config.global_config import GlobalConfig, PROJECT_ROOT
import logging
import shutil
import os


logger = logging.getLogger(__name__)
global_config = GlobalConfig()


class AbstractAnnabellRunner:

    def __init__(self, dataset_processor, a_run_environment_class):
        self.dataset_processor = dataset_processor
        self.run_environment = a_run_environment_class(self)

    def run(
        self,
    ):
        self.setup()
        self.run_processing()
        self.teardown()

    def setup(self):
        self.run_environment.setup()  # creates directories first
        self.run_environment.write_annabell_files()
        self.run_environment.copy_files_to_container_directory()

    def write_annabell_files_to_outputs_directory(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def teardown(self):
        self.run_environment.teardown()

    def copy_files_to_docker_directory(self):

        raise NotImplementedError("Subclasses should implement this method.")

    def copy_files_to_apptainer_directory(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def run_script(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def apptainer_command(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def docker_command(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def setup_apptainer_directories(self):
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
        self.run_environment.run_processing()

    def run_processing_docker(self):

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

    def run_processing_apptainer(self):
        # Use shlex.split to handle arguments safely
        command_args = shlex.split(self.apptainer_command())

        # Find the full path to apptainer executable
        apptainer_path = shutil.which("apptainer")
        if not apptainer_path:
            if global_config.platform_config.is_mac() and shutil.which("limactl"):
                command_args = ["limactl", "shell", "apptainer"] + command_args
            else:
                raise FileNotFoundError(
                    "'apptainer' command not found. Make sure Apptainer is installed and in your system's PATH."
                )

        # Replace 'apptainer' with full path in command_args if it's the first argument
        if apptainer_path and command_args[0] == "apptainer":
            command_args[0] = apptainer_path

        logger.info(f"Running Apptainer command: {' '.join(command_args)}")

        try:
            # Execute the command
            result = subprocess.run(
                command_args,
                check=True,
                text=True,
                capture_output=True,
                shell=False,
            )
            logger.info("STDOUT: %s", result.stdout)
            logger.info("STDERR: %s", result.stderr)
            logger.info("\nApptainer command executed successfully.")

        except subprocess.CalledProcessError as e:
            logger.critical(f"Apptainer command failed with exit code {e.returncode}")
            logger.error("STDOUT: %s", e.stdout)
            logger.error("STDERR: %s", e.stderr)
            raise

    def docker_runtime_weights_filepath(self):
        raise NotImplementedError("Subclasses should implement this method.")


class AnnabellTrainingRunner(AbstractAnnabellRunner):

    def setup(self):
        super().setup()

    def teardown(self):
        self.run_environment.copy_annabell_weights()
        self.run_environment.move_annabell_logfile()
        super().teardown()

    def setup_apptainer_directories(self):
        # Ensure the training directory exists inside the apptainer directory on the host
        training_dir = os.path.join(
            global_config.project_directory(),
            global_config.apptainer_training_directory(),
        )
        os.makedirs(training_dir, exist_ok=True)

    def apptainer_command(self):
        project_dir = global_config.project_directory()
        apptainer_dir = global_config.apptainer_directory()

        # Use relative paths from the apptainer working directory (/{apptainer_dir})
        # This matches the Docker pattern where Annabell uses relative paths like
        # "pre_training/file.txt" rather than absolute paths "/apptainer/pre_training/file.txt"
        training_dir = global_config.settings.file_locations.training_directory
        pre_training_dir = global_config.settings.file_locations.pre_training_directory
        log_filename = (
            global_config.settings.file_locations.annabell_log_training_filename
        )
        training_filename = global_config.training_filename()
        training_weights_filename = global_config.training_weights_filename()
        pre_training_weights_filename = global_config.pre_training_weights_filename()

        command = (
            f"apptainer exec --nv "
            f"--pwd /{apptainer_dir} "
            f"--bind {project_dir}/{apptainer_dir}/:/{apptainer_dir} "
            f"--bind {project_dir}/annabell_scripts/:/annabell_scripts "
            f"{project_dir}/{global_config.annabell_build_sif_filepath()} "
            f"/annabell_scripts/{self.run_script()} "
            f"{training_dir}/{log_filename} "
            f"{pre_training_dir}/{pre_training_weights_filename} "
            f"{training_dir}/{training_weights_filename} "
            f"{training_dir}/{training_filename} "
        )
        return command

    def run_script(self):
        return "train_annabell_squad.sh"

    def write_annabell_files_to_outputs_directory(self):

        self.dataset_processor.write_training_file(global_config.training_filepath())

    def copy_files_to_docker_directory(self):
        self.copy_file(
            global_config.training_filepath(),
            global_config.docker_training_directory(),
        )

    def copy_files_to_apptainer_directory(self):

        source = global_config.training_filepath()
        dest = global_config.apptainer_training_directory()
        dest_full = os.path.join(global_config.project_directory(), dest)
        os.makedirs(dest_full, exist_ok=True)
        self.copy_file(source, dest_full)
        # Verify the file arrived
        expected = os.path.join(dest_full, os.path.basename(source))
        if not os.path.exists(expected):
            raise FileNotFoundError(f"training file not found after copy: {expected}")
        logger.info(f"Verified training file exists at: {expected}")

    def docker_command(self):

        # <logfile> <pre-training_weights> <post-training_weights> <statements_file>

        command = (
            f"docker compose run --rm --remove-orphans --entrypoint ./{self.run_script()} app "
            f"{global_config.docker_runtime_training_log_filepath()} "
            f"{global_config.docker_runtime_pre_training_weights_filepath()} "
            f"{global_config.docker_runtime_training_weights_filepath()} "
            f"{global_config.docker_runtime_training_filepath()}"
        )
        return command

    def copy_annabell_weights_to_gdrive(self):
        # copy the pre-trained weights to the pre-training directory
        source_path = self.run_environment.training_weights_filepath()
        destination_path = os.path.join(
            global_config.training_directory(),
            global_config.training_weights_filename(),
        )

        self.copy_file(source_path, destination_path)

    def move_annabell_logfile_to_gdrive(self):
        source_path = self.run_environment.training_log_filepath()

        destination_path = global_config.training_log_filepath()

        self.move_file(source_path, destination_path)

    @staticmethod
    def apptainer_training_weights_filepath():
        return os.path.join(
            global_config.project_directory(),
            global_config.apptainer_training_weights_filepath(),
        )


class AnnabellPreTrainingRunner(AnnabellTrainingRunner):

    def setup(self):
        super().setup()

    def teardown(self):
        super().teardown()

    def run_script(self):
        return "pre_train_annabell_squad_nyc.sh"

    def setup_apptainer_directories(self):
        # Ensure the pre_training directory exists inside the apptainer directory on the host
        pre_training_dir = os.path.join(
            global_config.project_directory(),
            global_config.apptainer_pre_training_directory(),
        )
        os.makedirs(pre_training_dir, exist_ok=True)

    def apptainer_command(self):
        project_dir = global_config.project_directory()
        apptainer_dir = global_config.apptainer_directory()

        # Use relative paths from the apptainer working directory (/{apptainer_dir})
        # This matches the Docker pattern where Annabell uses relative paths like
        # "pre_training/file.txt" rather than absolute paths "/apptainer/pre_training/file.txt"
        pre_training_dir = global_config.settings.file_locations.pre_training_directory
        log_filename = (
            global_config.settings.file_locations.annabell_log_pre_training_filename
        )
        training_filename = global_config.pre_training_filename()
        weights_filename = global_config.pre_training_weights_filename()

        command = (
            f"apptainer exec --nv "
            f"--pwd /{apptainer_dir} "
            f"--bind {project_dir}/{apptainer_dir}/:/{apptainer_dir} "
            f"--bind {project_dir}/annabell_scripts/:/annabell_scripts "
            f"{project_dir}/{global_config.annabell_build_sif_filepath()} "
            f"/annabell_scripts/{self.run_script()} "
            f"{pre_training_dir}/{log_filename} "
            f"{pre_training_dir}/{training_filename} "
            f"{pre_training_dir}/{weights_filename}"
        )
        return command

    def docker_command(self):

        command = (
            f"docker compose run --rm --remove-orphans --entrypoint ./{self.run_script()} app "
            f"{global_config.docker_runtime_pre_training_log_filepath()} "
            f"{global_config.docker_runtime_pre_training_filepath()} "
            f"{global_config.docker_runtime_pre_training_weights_filepath()}"
        )
        return command

    def write_annabell_files_to_outputs_directory(self):

        self.dataset_processor.write_pretraining_file(
            global_config.pre_training_filepath(), global_config.auto_save_weights()
        )

    def copy_files_to_docker_directory(self):
        self.copy_file(
            global_config.pre_training_filepath(),
            global_config.docker_pre_training_directory(),
        )

    def copy_files_to_apptainer_directory(self):

        source = global_config.pre_training_filepath()
        dest = global_config.apptainer_pre_training_directory()
        dest_full = os.path.join(global_config.project_directory(), dest)
        os.makedirs(dest_full, exist_ok=True)
        self.copy_file(source, dest_full)
        # Verify the file arrived
        expected = os.path.join(dest_full, os.path.basename(source))
        if not os.path.exists(expected):
            raise FileNotFoundError(
                f"Pre-training file not found after copy: {expected}"
            )
        logger.info(f"Verified pre-training file exists at: {expected}")

    def docker_runtime_weights_filepath(self):
        return global_config.docker_runtime_pre_training_weights_filepath()

    @staticmethod
    def apptainer_pre_training_weights_filepath():
        return os.path.join(
            global_config.project_directory(),
            global_config.apptainer_pre_training_weights_filepath(),
        )

    def copy_annabell_weights_to_gdrive(self):
        # copy the pre-trained weights to the pre-training directory

        source_path = self.run_environment.pre_training_weights_filepath()
        destination_path = os.path.join(
            global_config.pre_training_directory(),
            global_config.pre_training_weights_filename(),
        )

        self.copy_file(source_path, destination_path)

    def move_annabell_logfile_to_gdrive(self):
        source_path = os.path.join(
            self.run_environment.pre_training_directory(),
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

    def teardown(self):
        self.run_environment.move_annabell_logfile()
        super().teardown()

    def docker_command(self):
        # <logfile> <pre-training_weights> <testing_file>
        command = (
            f"docker compose run --rm --entrypoint ./{self.run_script()} app "
            f"{self.docker_runtime_log_filepath()} "
            f"{self.docker_runtime_weights_filepath()} "
            f"{self.docker_runtime_commands_filepath()}"
        )
        return command

    def docker_runtime_weights_filepath(self):
        return global_config.docker_runtime_training_weights_filepath()

    def docker_runtime_commands_filepath(self):
        return global_config.docker_runtime_testing_filepath()

    def docker_runtime_log_filepath(self):
        return global_config.docker_runtime_testing_log_filepath()

    def copy_files_to_docker_directory(self):
        self.copy_file(
            global_config.testing_filepath(),
            global_config.docker_testing_filepath(),
        )

    def write_annabell_files_to_outputs_directory(self):

        self.dataset_processor.write_testing_file(global_config.testing_filepath())

    def move_annabell_logfile_to_gdrive(self):

        source_path = self.run_environment.testing_log_filepath()

        destination_path = global_config.testing_log_filepath()

        self.move_file(source_path, destination_path)

    def setup_apptainer_directories(self):
        # Ensure the training directory exists inside the apptainer directory on the host
        testing_dir = os.path.join(
            global_config.project_directory(),
            global_config.apptainer_testing_directory(),
        )
        os.makedirs(testing_dir, exist_ok=True)

    def copy_files_to_apptainer_directory(self):
        # Copy the testing file
        source_commands = global_config.testing_filepath()
        dest_commands = global_config.apptainer_testing_filepath()
        dest_commands_full = os.path.join(
            global_config.project_directory(), dest_commands
        )
        os.makedirs(os.path.dirname(dest_commands_full), exist_ok=True)
        self.copy_file(source_commands, dest_commands_full)

        # Copy the training weights file
        source_weights = global_config.training_weights_filepath()
        dest_weights = global_config.apptainer_training_weights_filepath()
        dest_weights_full = os.path.join(
            global_config.project_directory(), dest_weights
        )
        os.makedirs(os.path.dirname(dest_weights_full), exist_ok=True)
        self.copy_file(source_weights, dest_weights_full)

    def apptainer_command(self):
        project_dir = global_config.project_directory()
        apptainer_dir = global_config.apptainer_directory()

        # Use relative paths from the apptainer working directory (/{apptainer_dir})
        testing_dir = global_config.settings.file_locations.testing_directory
        training_dir = global_config.settings.file_locations.training_directory
        log_filename = (
            global_config.settings.file_locations.annabell_log_testing_filename
        )
        commands_filename = global_config.settings.file_locations.testing_filename
        weights_filename = global_config.training_weights_filename()

        command = (
            f"apptainer exec --nv "
            f"--pwd /{apptainer_dir} "
            f"--bind {project_dir}/{apptainer_dir}/:/{apptainer_dir} "
            f"--bind {project_dir}/annabell_scripts/:/annabell_scripts "
            f"{project_dir}/{global_config.annabell_build_sif_filepath()} "
            f"/annabell_scripts/{self.run_script()} "
            f"{testing_dir}/{log_filename} "
            f"{training_dir}/{weights_filename} "
            f"{testing_dir}/{commands_filename}"
        )
        return command


class AnnabellPreTrainingTestingRunner(AnnabellTestingRunner):

    def setup(self):
        super().setup()

    def teardown(self):
        super().teardown()

    def setup_apptainer_directories(self):
        # Ensure the training directory exists inside the apptainer directory on the host
        testing_dir = os.path.join(
            global_config.project_directory(),
            global_config.apptainer_testing_directory(),
        )
        os.makedirs(testing_dir, exist_ok=True)

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

    def write_annabell_files_to_outputs_directory(self):
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
        source_path = (
            self.run_environment.pre_training_validation_testing_log_filepath()
        )
        destination_path = global_config.pre_training_validation_testing_log_filepath()

        self.move_file(source_path, destination_path)

    def copy_files_to_apptainer_directory(self):
        # Copy the pre-training validation/testing file
        source_commands = global_config.pre_training_validation_testing_filepath()
        dest_commands = (
            global_config.apptainer_pre_training_validation_testing_filepath()
        )
        dest_commands_full = os.path.join(
            global_config.project_directory(), dest_commands
        )
        os.makedirs(os.path.dirname(dest_commands_full), exist_ok=True)
        self.copy_file(source_commands, dest_commands_full)

        # Copy the pre-training weights file
        source_weights = global_config.pre_training_weights_filepath()
        dest_weights = global_config.apptainer_pre_training_weights_filepath()
        dest_weights_full = os.path.join(
            global_config.project_directory(), dest_weights
        )
        os.makedirs(os.path.dirname(dest_weights_full), exist_ok=True)
        self.copy_file(source_weights, dest_weights_full)

    def apptainer_command(self):
        project_dir = global_config.project_directory()
        apptainer_dir = global_config.apptainer_directory()

        # Use relative paths from the apptainer working directory (/{apptainer_dir})
        pre_training_validation_testing_dir = (
            global_config.settings.file_locations.testing_directory
        )
        pre_training_dir = global_config.settings.file_locations.pre_training_directory
        log_filename = (
            global_config.settings.file_locations.annabell_log_pre_training_validation_testing_filename
        )
        commands_filename = (
            global_config.settings.file_locations.pre_training_validation_testing_filename
        )
        weights_filename = global_config.pre_training_weights_filename()

        command = (
            f"apptainer exec --nv "
            f"--pwd /{apptainer_dir} "
            f"--bind {project_dir}/{apptainer_dir}/:/{apptainer_dir} "
            f"--bind {project_dir}/annabell_scripts/:/annabell_scripts "
            f"{project_dir}/{global_config.annabell_build_sif_filepath()} "
            f"/annabell_scripts/{self.run_script()} "
            f"{pre_training_validation_testing_dir}/{log_filename} "
            f"{pre_training_dir}/{weights_filename} "
            f"{pre_training_validation_testing_dir}/{commands_filename}"
        )
        return command
