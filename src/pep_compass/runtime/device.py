"""Runtime device inspection and preflight validation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TorchRuntime:
    """Describe the installed Torch build and available CUDA runtime."""

    version: str
    compiled_cuda: str | None
    cuda_available: bool
    cuda_device_count: int


def inspect_torch_runtime() -> TorchRuntime:
    """Inspect Torch without allocating model tensors.

    :return: Installed Torch version and CUDA availability.
    """
    import torch

    cuda_available = torch.cuda.is_available()
    return TorchRuntime(
        version=torch.__version__,
        compiled_cuda=torch.version.cuda,
        cuda_available=cuda_available,
        cuda_device_count=torch.cuda.device_count() if cuda_available else 0,
    )


def validate_execution_device(device: str, runtime: TorchRuntime) -> None:
    """Reject an unavailable configured execution device before model loading.

    :param device: Torch device configured for the autoencoder.
    :param runtime: Inspected Torch runtime.
    :raises RuntimeError: If CUDA was requested but cannot be used.
    """
    if not device.startswith("cuda"):
        return
    if runtime.compiled_cuda is None:
        raise RuntimeError(
            f"autoencoder.device={device!r} requires a CUDA Torch build, but "
            f"torch {runtime.version} was compiled without CUDA. Run with a "
            "CUDA extra, for example `uv run --extra cu118 pep-compass ...`."
        )
    if not runtime.cuda_available:
        raise RuntimeError(
            f"autoencoder.device={device!r} was requested and torch "
            f"{runtime.version} was compiled for CUDA {runtime.compiled_cuda}, "
            "but CUDA is unavailable. Check the NVIDIA driver and container/WSL "
            "GPU access."
        )

