"""Provide logging without importing the host application."""
import logging


def get_custom_logger(name: str) -> logging.Logger:
    """Return the logger for a library module.

    :param str name: Fully qualified module name.
    :return: Logger configured by the calling application.
    :rtype: logging.Logger
    """
    return logging.getLogger(name)
