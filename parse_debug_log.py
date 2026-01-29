import sys
import argparse


class LogFileParser:
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        # Buffers to track
        self.buffers = [
            "InPhB",
            "WkPhB",
            "WGB",
            "StartPh",
            "CurrStartPh",
            "GoalWG",
            "GoalPh",
        ]
        # Actions to track
        self.actions = ["Next acquisition action", "Next elaboration action"]

        # Initialize state as None for all tracked items
        self.current_state = {k: None for k in self.buffers + self.actions}

    def parse(self):
        try:
            with open(self.log_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    # 1. Check for Command
                    if line.startswith("."):
                        if line.startswith(".monitor"):
                            continue
                        print(f"\nCommand: {line}")
                        continue

                    # Check for Operating Mode Change
                    if "Operating mode changed to:" in line:
                        print(f"\n{line}")
                        continue

                    # General key-value parsing
                    if ":" in line:
                        key, value = line.split(":", 1)
                        key = key.strip()
                        value = value.strip()

                        if key in self.buffers:
                            if self.current_state[key] != value:
                                print(f"  {key}: {value}")
                                self.current_state[key] = value
                            continue

                        if key in self.actions:
                            if self.current_state[key] != value:
                                print(f"  {key}: {value}")
                                self.current_state[key] = value
                            continue

        except FileNotFoundError:
            print(f"Error: File not found at {self.log_file_path}")
            sys.exit(1)
        except Exception as e:
            print(f"An error occurred: {e}")
            sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse specific activity from a log file."
    )
    parser.add_argument(
        "logfile",
        nargs="?",
        default="docker/shared_data/debug/debug_example_log.txt",
        help="Path to the log file",
    )

    args = parser.parse_args()

    log_parser = LogFileParser(args.logfile)
    log_parser.parse()