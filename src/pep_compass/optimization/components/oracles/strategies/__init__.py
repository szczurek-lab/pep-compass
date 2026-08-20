"""Built-in lazily constructed oracle strategies."""

from importlib import import_module
from typing import Any

from pep_compass.optimization.components.oracles.manager import OracleManager
from pep_compass.optimization.components.oracles.model_registry import (
    register_oracle_models,
)
from pep_compass.optimization.components.oracles.strategies.black_box import BlackBoxOracle
from pep_compass.utils.strategy_factory import parameter_contract

_COMMON = {
    "batch_size", "parallelize", "num_workers", "evaluation_budget",
    "force_isolation", "evaluation_batch_size",
}

register_oracle_models()


def _black_box_oracle(
    module_name: str,
    class_name: str,
    method: str,
    parameters: dict[str, Any],
) -> BlackBoxOracle:
    """Construct one black-box adapter without eagerly importing its model."""
    adapter_batch_size = parameters.pop("evaluation_batch_size", None)
    module = import_module(module_name)
    black_box_type = getattr(module, class_name)
    return BlackBoxOracle(
        black_box_type(**parameters),
        field_name=f"oracle.{method}.score",
        batch_size=adapter_batch_size,
    )


@OracleManager.register("apex")
@parameter_contract(
    accepted=_COMMON
    | {"mic_aggregate", "mic_bacteria", "model", "device", "models_directory"}
)
def build_apex(**parameters: Any) -> BlackBoxOracle:
    """Build the APEX oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.apex.oracle",
        "APEXBlackBox",
        "apex",
        parameters,
    )


@OracleManager.register("apex_original")
@parameter_contract(
    accepted=_COMMON
    | {"mic_aggregate", "mic_bacteria", "model", "device", "models_directory"}
)
def build_apex_original(**parameters: Any) -> BlackBoxOracle:
    """Build the preserved pre-refactor APEX oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.apex_original.oracle",
        "APEXBlackBox",
        "apex_original",
        parameters,
    )


@OracleManager.register("battleamp")
@parameter_contract(accepted=_COMMON | {"device", "model", "models_directory"})
def build_battleamp(**parameters: Any) -> BlackBoxOracle:
    """Build the BattleAMP oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.battleamp.oracle",
        "BattleAMPBlackBox",
        "battleamp",
        parameters,
    )


@OracleManager.register("eipred")
@parameter_contract(
    accepted=_COMMON | {"mic_aggregate", "model", "models_directory"}
)
def build_eipred(**parameters: Any) -> BlackBoxOracle:
    """Build the EIPred oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.eipred.oracle",
        "EIPredBlackBox",
        "eipred",
        parameters,
    )


@OracleManager.register("hydrophobicity")
@parameter_contract(accepted=_COMMON | {"scale"})
def build_hydrophobicity(**parameters: Any) -> BlackBoxOracle:
    """Build the hydrophobicity oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.hydrophobicity.oracle",
        "HydrophobicityBlackBox",
        "hydrophobicity",
        parameters,
    )


@OracleManager.register("mbc_attention")
@parameter_contract(accepted=_COMMON | {"device", "model", "models_directory"})
def build_mbc_attention(**parameters: Any) -> BlackBoxOracle:
    """Build the MBC-Attention oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.mbc_attention.oracle",
        "MBCAttentionBlackBox",
        "mbc_attention",
        parameters,
    )


@OracleManager.register("toxipep")
@parameter_contract(accepted=_COMMON | {"device", "model", "models_directory"})
def build_toxipep(**parameters: Any) -> BlackBoxOracle:
    """Build the ToxiPep oracle strategy."""
    return _black_box_oracle(
        "pep_compass.optimization.components.oracles.strategies.toxipep.oracle",
        "ToxiPepBlackBox",
        "toxipep",
        parameters,
    )


__all__ = ["BlackBoxOracle"]
