"""Logging configuration helper shared by CLI entrypoints."""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path


def configure_logging(
    verbose: bool = False,
    very_verbose: bool = False,
    log_to_file: bool | Path = True,
    log_file_prepend: str = "HDF5Translator_",
) -> None:
    """Configure root logging to stdout and an optional timestamped file."""

    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    log_datefmt = "%Y-%m-%d %H:%M:%S"
    if very_verbose:
        level = logging.DEBUG
    elif verbose:
        level = logging.INFO
    else:
        level = logging.WARNING

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_to_file:
        if isinstance(log_to_file, Path):
            log_filename = log_to_file
        else:
            log_filename = Path(f"{log_file_prepend}{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        handlers.append(logging.FileHandler(log_filename))

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt=log_datefmt,
        handlers=handlers,
    )
