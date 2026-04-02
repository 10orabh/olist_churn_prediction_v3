import logging
import os 
from logging.handlers import RotatingFileHandler
from from_root import from_root
from datetime import datetime

# constants for log configuration 
LOG_DIR = 'logs'
LOG_FILE = f'{datetime.now().strftime("%m_%d_%Y_%H_%M_%S")}.log'
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 5 #Number of backup log files to keep

# construct log file path
log_file_path = os.path.join(from_root(), LOG_DIR)
os.makedirs(log_file_path, exist_ok=True) # create log directory if it doesn't exist
log_file_path = os.path.join(log_file_path, LOG_FILE)


def configure_logger():
    """Configures the logger with a rotating file handler."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set the logging level to DEBUG

    # Create a rotating file handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT)
    formatter = logging.Formatter("[%(asctime)s] %(name)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)  # Set console logging level to INFO

    # Add the handler to the logger
    logger.addHandler(file_handler) 
    logger.addHandler(console_handler) 

# Call the configure_logger function to set up logging
configure_logger()
