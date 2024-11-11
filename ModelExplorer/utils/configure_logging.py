import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import Union

def configure_logging(verbose: bool = False, very_verbose: bool = False, log_to_file: Union[bool, Path] = True, log_file_prepend: str = "HDF5Translator_"):
    """
    Configure logging to output to stdout and an optional log file.
    
    Args:
        verbose (bool): Enable verbose logging output.
        very_verbose (bool): Enable very verbose logging output.
        log_to_file (bool, Path): Enable logging to a file. If a Path object is provided, the log file will be written to that path.
        log_file_prepend (str): Prefix to prepend to the log file name.
    """
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    

    if very_verbose: 
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        if isinstance(log_to_file, Path):
            log_filename = log_to_file
        else:
            log_filename = f"{log_file_prepend}{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        handlers.append(logging.FileHandler(log_filename))

    # Configure root logger
    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=log_datefmt,
        handlers=handlers,
    )
