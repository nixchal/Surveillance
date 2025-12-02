"""Console entry point for the Campus Surveillance system."""

from __future__ import annotations

import logging
import signal
import sys
from contextlib import contextmanager

import config
from src.detector import CampusSurveillance


LOGGER = logging.getLogger(__name__)


@contextmanager
def graceful_shutdown(detector: CampusSurveillance):
    """Ensure the detector releases resources on shutdown."""

    def _signal_handler(signum, frame):
        LOGGER.info("Received signal %s. Stopping surveillance...", signum)
        detector.stop()

    original_handlers = {
        sig: signal.getsignal(sig)
        for sig in (signal.SIGINT, signal.SIGTERM)
    }

    for sig in original_handlers:
        signal.signal(sig, _signal_handler)

    try:
        yield
    finally:
        LOGGER.info("Shutting down Campus Surveillance system")
        detector.stop()
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)


def main() -> int:
    config.configure_logging()

    LOGGER.info("Starting Campus Surveillance system...")
    detector = CampusSurveillance()

    with graceful_shutdown(detector):
        detector.run()

    LOGGER.info("Surveillance session finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())

