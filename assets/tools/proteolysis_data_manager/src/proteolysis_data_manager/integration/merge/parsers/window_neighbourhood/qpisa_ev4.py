"""Parse qPISA EV4 into the common measurement contract."""

from proteolysis_data_manager.integration.merge.parsers.window_neighbourhood._qpisa_endpoint import (
    parse_experiment,
)


def parse(root, *, data=None):
    """Return all EV4 observations with source inputs and available windows.

    :param root: Acquired window-neighbourhood series directory.
    :param data: Optional endpoint measurement frame.
    :return: Validated common dataset.
    """
    return parse_experiment(root, "EV4", data=data)
