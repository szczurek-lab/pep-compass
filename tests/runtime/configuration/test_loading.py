"""Tests for typed YAML runtime configuration loading."""

from pathlib import Path

from pep_compass.runtime.configuration import load_runtime_configuration
import pytest


def test_minimal_yaml_loads_into_typed_configuration() -> None:
    """A valid YAML document must produce explicit configuration sections."""
    path = Path("tests/mock_data/configurations/minimal.yaml")

    configuration = load_runtime_configuration(path)

    assert configuration.experiment.name == "mock"
    assert configuration.autoencoder.method == "hydramp"
    assert configuration.autoencoder.model == "article_25"
    assert configuration.execution.backend == "local"


def test_loading_rejects_unknown_named_autoencoder_model(tmp_path) -> None:
    """Configuration validation must fail without loading model weights."""
    path = tmp_path / "invalid.yaml"
    path.write_text(
        """
experiment:
  name: invalid
  input:
    sequences: [AA]
autoencoder:
  method: hydramp
  model: missing
  parameters:
    jacobian_eps: 0.001
    field_eps: 0.001
pipeline:
  steps: []
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unknown autoencoder model"):
        load_runtime_configuration(path)


def test_reference_lebo_uses_reference_local_enumeration_parameters() -> None:
    """The maintained reference config must retain origin/dev baseline values."""
    configuration = load_runtime_configuration(
        "assets/experiments/configs/reference_lebo.yaml"
    )
    outer_loop = configuration.pipeline["steps"][0]["loop"]
    local = outer_loop["steps"][0]["local_enumeration"]
    walker = local["walker"]["parameters"]

    assert configuration.autoencoder.parameters["jacobian_eps"] == 0.05
    assert configuration.autoencoder.parameters["field_eps"] == 0.05
    assert local["trajectories"] == 10
    assert local["walk_time"] == 0.1
    assert walker["geometry"]["parameters"]["kappa"] == 0.01
    assert walker["position_update"]["parameters"] == {
        "epsilon": 0.1,
        "delta_max": 0.5,
    }
