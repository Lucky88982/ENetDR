"""Logging utilities."""
import os
import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger('enetdr')

def setup_file_logger(output_dir, log_name="train.log"):
    """Mirror console logs to a file under the experiment output directory."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(output_dir, log_name))

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if getattr(handler, "_experiment_log_path", None) == log_path:
            return log_path

    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        fmt='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S'
    ))
    file_handler._experiment_log_path = log_path
    root_logger.addHandler(file_handler)
    logger.info(f"Logging to file: {log_path}")
    return log_path
