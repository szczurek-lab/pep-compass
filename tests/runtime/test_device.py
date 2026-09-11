"""Tests for explicit Torch device preflight validation."""

import pytest

from pep_compass.runtime.backends.subprocess import worker_command
from pep_compass.runtime.cli import _parser
from pep_compass.runtime.device import TorchRuntime, validate_execution_device


def test_cpu_device_accepts_cpu_only_torch() -> None:
    runtime = TorchRuntime("2.10.0+cpu", None, False, 0)

    validate_execution_device("cpu", runtime)


def test_cuda_device_rejects_cpu_only_torch() -> None:
    runtime = TorchRuntime("2.10.0+cpu", None, False, 0)

    with pytest.raises(RuntimeError, match="compiled without CUDA"):
        validate_execution_device("cuda", runtime)


def test_cuda_device_rejects_unavailable_driver() -> None:
    runtime = TorchRuntime("2.5.1+cu118", "11.8", False, 0)

    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        validate_execution_device("cuda:0", runtime)


def test_cuda_device_accepts_available_cuda_runtime() -> None:
    runtime = TorchRuntime("2.5.1+cu118", "11.8", True, 1)

    validate_execution_device("cuda", runtime)


def test_cli_accepts_explicit_autoencoder_device_override() -> None:
    arguments = _parser().parse_args(
        ["test-run", "experiment.yaml", "--device", "cuda:0"]
    )

    assert arguments.autoencoder_device == "cuda:0"


def test_subprocess_worker_preserves_device_override(tmp_path) -> None:
    command = worker_command(
        tmp_path / "experiment.yaml",
        3,
        working_directory=tmp_path,
        autoencoder_device="cuda",
    )

    assert command[-2:] == ["--device", "cuda"]
