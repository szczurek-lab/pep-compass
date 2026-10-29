"""Parse the dramp source into the common contract."""

from proteolysis_data_manager.integration.merge.parsers.half_life._half_life import (
    parse_source,
)


def parse(root, *, data=None):
    """Return dramp observations using the shared quantity/condition contract.

    :param root: Acquired half-time series directory.
    :param data: Optional complete existing adapter inventory.
    :return: Validated unified dataset.
    """
    return parse_source(root, "dramp", data=data)
