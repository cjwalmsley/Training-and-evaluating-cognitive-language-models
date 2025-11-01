import os
from datetime import datetime


class LogWriter:

    if os.path.exists("/Volumes/X9 Pro/logs"):
        log_directory = "/Volumes/X9 Pro/logs"
    if os.path.exists("/Users/chris/logs"):
        log_directory = "/Users/chris/logs"
    else:
        log_directory = "/home/chris/logs"

    def __init__(self, log_filename):
        self.log_filepath = os.path.join(LogWriter.log_directory, log_filename)
        if not os.path.exists(LogWriter.log_directory):
            os.makedirs(LogWriter.log_directory)
        if not os.path.exists(self.log_filepath):
            self.write_log_header(self.log_filepath)

    def log(self, message):
        with open(self.log_filepath, "a") as file:
            file.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\t{message}\n")

    def write_log_header(self, filename):
        with open(filename, "w") as file:
            file.write("timestamp\tmessage\n")
        print("Log file created")