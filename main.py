import logging
from src.logging_utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info("Flood monitoring project entrypoint invoked")
    print("Hello from flood-monitoring!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Unhandled exception in main: %s", str(e))
        raise
