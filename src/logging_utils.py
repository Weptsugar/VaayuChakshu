import logging
import os
from datetime import datetime


def setup_logging(log_level: int = logging.INFO, log_dir: str | None = None, log_file: str | None = None):
    """Configure root logger with console and optional file handlers.

    Parameters
    ----------
    log_level: int
        Logging level for root logger. Defaults to ``logging.INFO``.
    log_dir: str | None
        Folder where log file should be saved. Created if it does not exist.
        If *None*, a ``logs`` directory is created in the project root.
    log_file: str | None
        Log filename. If *None*, a timestamped filename is generated.
    """

    # Ensure we only configure logging once in interactive / multi-process settings.
    if getattr(setup_logging, "_is_configured", False):  # type: ignore[attr-defined]
        return

    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter(
        fmt="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir is None:
        log_dir = os.path.join(os.getcwd(), "logs")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    if log_file is None:
        log_file = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_path = os.path.join(log_dir, log_file)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    logger.debug("Logging configured. Logs will be written to %s", file_path)

    # Mark as configured
    setup_logging._is_configured = True  # type: ignore[attr-defined] 