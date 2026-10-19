"""Instantiate the same public Python constructors selected by Hydra targets."""

from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.utils import instantiate

from pep_compass.core.wrapper import Pipeline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pep_compass.autoencoder.base import Autoencoder
    from pep_compass.autoencoder.resource import AutoencoderResource
    from pep_compass.data_schemas.execution import ExecutionSettings


def create_pipeline(
    configuration: Mapping[str, object],
    *,
    autoencoder: AutoencoderResource | Autoencoder | None = None,
    execution: ExecutionSettings | None = None,
) -> Pipeline:
    """Construct lightweight wrapper declarations; validation never loads weights.

    :param configuration: Hydra mapping with a public pipeline or preset _target_.
    :param autoencoder: Optional already initialized Python resource.
    :param execution: Optional runner initialization defaults.
    :return: Editable Pipeline using exactly the Python component API.

    :type configuration: Mapping[str, object]
    :type autoencoder: AutoencoderResource | Autoencoder | None
    :type execution: ExecutionSettings | None
    :rtype: Pipeline
    """
    pipeline = instantiate(configuration, _convert_="all")
    if not isinstance(pipeline, Pipeline):
        raise TypeError("pipeline._target_ must construct a Pipeline.")
    if autoencoder is not None:
        pipeline.add_autoencoder(autoencoder)
    pipeline.set_runner_settings(execution=execution)
    pipeline.validate()
    return pipeline
