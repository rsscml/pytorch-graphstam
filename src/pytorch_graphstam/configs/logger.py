import logging
import os

class Logger:
    """
    Create logger that writes to a file and to stdout.
    """

    def __init__(
        self,
        log_file: str,
        log_level: str = "DEBUG",
        console_level: str = "INFO",
    ) -> None:
        """
        Initialize the Logger.

        Args:
            log_file (str): Path to the log file.
            log_level (int): Logging level for the file handler. Default is DEBUG.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

        # Check if handlers already exist
        if self.logger.hasHandlers():
            self.logger.handlers.clear()

        # Create a file handler
        try:
            if not os.path.exists(os.path.dirname(log_file)):
                os.makedirs(os.path.dirname(log_file))
            handler = logging.FileHandler(log_file)
            handler.setLevel(log_level)
        except OSError as e:
            print(f"Error creating log file: {e}")
            return

        # Create a logging format
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)

        # Add the file handler to the logger
        self.logger.addHandler(handler)

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)

    def get_logger(self) -> logging.Logger:
        """
        Get the configured logger instance.

        Returns:
            logging.Logger: The configured logger instance.
        """
        return self.logger