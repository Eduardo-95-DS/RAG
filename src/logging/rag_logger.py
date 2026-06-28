"""Structured logging for the RAG pipeline."""
import logging
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str = "rag") -> logging.Logger:
    """
    Return a logger that writes to both stdout and a rotating file (rag.log).
    Safe to call multiple times — handlers are only added once.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # stdout (captured by Streamlit Cloud)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    # rotating file — 1 MB max, keep 3 backups
    log_path = Path("rag.log")
    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger
