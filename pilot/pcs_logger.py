import logging
import os


class PilotComputeServiceLogger:
    """
    Singleton logger class for Pilot Compute Service.

    This class provides a singleton logger instance for logging messages to both a file and the console.
    The log file location can be customized using the `PILOT_LOG_FILE` environment variable. If not provided,
    the log file will be created in the user's home directory with the default name `pilot-quantum.log`.
    """

    _instance = None

    def __new__(cls, pcs_working_directory):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            cls._instance.pcs_working_directory = pcs_working_directory
        return cls._instance

    def __init__(self, pcs_working_directory):
        if not self._initialized:
            log_file = os.path.join(pcs_working_directory, "pilot-quantum.log")
            log_level = logging.DEBUG

            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(log_level)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

            # Log to file
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

            # Log to console
            console_handler = logging.StreamHandler()
            console_handler.setLevel(log_level)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

            self._initialized = True

    def log(self, message, level=logging.INFO):
        if level == logging.INFO:
            self.logger.info(message)
        elif level == logging.WARNING:
            self.logger.warning(message)
        elif level == logging.ERROR:
            self.logger.error(message)
        elif level == logging.CRITICAL:
            self.logger.critical(message)
        else:
            self.logger.debug(message)

    def info(self, message):
        self.log(message, logging.INFO)

    def warning(self, message):
        self.log(message, logging.WARNING)

    def error(self, message):
        self.log(message, logging.ERROR)

    def critical(self, message):
        self.log(message, logging.CRITICAL)

    def debug(self, message):
        self.log(message, logging.DEBUG)


# Example usage:
if __name__ == "__main__":
    pcs_working_directory = "/path/to/pcs_working_directory"
    logger1 = PilotComputeServiceLogger(pcs_working_directory)
    logger2 = PilotComputeServiceLogger(pcs_working_directory)

    # Both logger1 and logger2 are the same instance
    print(logger1 is logger2)  # Output: True

    logger1.info("This is an info message")
    logger2.warning("This is a warning message")
