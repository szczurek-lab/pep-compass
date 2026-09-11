"""Library logging helpers."""

import logging


def get_custom_logger(name: str) -> logging.Logger:
    """Return the library logger for a module.

    :param name: Fully qualified module name.
    :type name: str
    :return: Logger associated with ``name``.
    :rtype: logging.Logger
    """
    return logging.getLogger(name)
