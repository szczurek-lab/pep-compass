"""Named APEX ensemble registry and weight discovery."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pep_compass.optimization.components.oracles.model_registry import (
    resolve_oracle_model,
)


_DEFAULT_PATHOGENS = (
    "A. baumannii ATCC 19606",
    "E. coli ATCC 11775",
    "E. coli AIG221",
    "E. coli AIG222",
    "K. pneumoniae ATCC 13883",
    "P. aeruginosa PA01",
    "P. aeruginosa PA14",
    "S. aureus ATCC 12600",
    "S. aureus (ATCC BAA-1556) - MRSA",
    "vancomycin-resistant E. faecalis ATCC 700802",
    "vancomycin-resistant E. faecium ATCC 700221",
)

_FULL_PATHOGENS = _DEFAULT_PATHOGENS + (
    "A. muciniphila ATCC BAA-835",
    "B. fragilis ATCC25285",
    "B. vulgatus ATCC8482",
    "C. aerofaciens ATCC25986",
    "C. scindens ATCC35704",
    "B. thetaiotaomicron ATCC29148",
    "B. thetaiotaomicron Complemmented",
    "B. thetaiotaomicron Mutant",
    "B. uniformis ATCC8492",
    "B. eggerthi ATCC27754",
    "C. spiroforme ATCC29900",
    "P. distasonis ATCC8503",
    "P. copri DSMZ18205",
    "B. ovatus ATCC8483",
    "E. rectale ATCC33656",
    "C. symbiosum",
    "R. obeum",
    "R. torques",
    "E. coli Nissle",
    "Salmonella enterica ATCC 9150 (BEIRES NR-515)",
    "Salmonella enterica (BEIRES NR-170)",
    "Salmonella enterica ATCC 9150 (BEIRES NR-174)",
    "L. monocytogenes ATCC 19111 (BEIRES NR-106)",
)


@dataclass(frozen=True, slots=True)
class APEXModelDescriptor:
    """Describe one complete named APEX weight ensemble."""

    directory: str
    pattern: str
    expected_models: int
    pathogens: tuple[str, ...]
    download_script: str


APEX_MODELS = {
    "default": APEXModelDescriptor(
        "default",
        "APEX_*",
        8,
        _DEFAULT_PATHOGENS,
        "assets/scripts/downloads/apex/download_apex_models_default.sh",
    ),
    "full": APEXModelDescriptor(
        "full",
        "trained_*",
        40,
        _FULL_PATHOGENS,
        "assets/scripts/downloads/apex/download_apex_models_full.sh",
    ),
}


def resolve_apex_weights(
    model: str,
    models_directory: str | Path | None = None,
) -> tuple[APEXModelDescriptor, tuple[Path, ...]]:
    """Resolve and validate every file in a named APEX ensemble.

    :param model: Registered ensemble name.
    :param models_directory: Optional root replacing the packaged ``models`` root.
    :return: Model descriptor and deterministically ordered weight paths.
    :raises ValueError: If ``model`` is not registered.
    :raises FileNotFoundError: If its weight set is absent or incomplete.
    """
    try:
        descriptor = APEX_MODELS[model]
    except KeyError as error:
        raise ValueError(
            f"Unknown APEX model: {model!r}. Available models: {sorted(APEX_MODELS)}."
        ) from error
    _, paths = resolve_oracle_model(
        "apex",
        model,
        models_directory=models_directory,
    )
    return descriptor, paths
