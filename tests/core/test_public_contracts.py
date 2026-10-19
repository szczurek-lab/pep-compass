"""Structural acceptance checks complementing execution and numerical tests."""

import ast
import inspect
from pathlib import Path

import pep_compass.steps as ps  # ps - PipelineSteps
from pep_compass import Pipeline
from pep_compass.autoencoder import AutoencoderResource, HydrAMP
from pep_compass.autoencoder.base import Autoencoder
from pep_compass.data_schemas.execution import StorageSettings


def source_root():
    import pep_compass

    return Path(pep_compass.__file__).parent


def test_every_source_function_declares_parameter_and_return_types():
    missing = []
    for path in source_root().rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.returns is None:
                missing.append((path, node.lineno, "return"))
            arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
            arguments += [
                value
                for value in (node.args.vararg, node.args.kwarg)
                if value is not None
            ]
            for argument in arguments:
                if argument.arg not in {"self", "cls"} and argument.annotation is None:
                    missing.append((path, node.lineno, argument.arg))
    assert missing == []


def test_removed_configuration_paths_and_dynamic_root_exports_do_not_return():
    root = source_root()
    assert not (root / "configuration").exists()
    assert not (root / "autoencoder/hydramp.py").exists()
    assert not (root / "autoencoder/factory.py").exists()
    tree = ast.parse((root / "__init__.py").read_text())
    assert not any(
        isinstance(node, ast.FunctionDef) and node.name == "__getattr__"
        for node in tree.body
    )
    assert HydrAMP.__module__.endswith("strategies.hydramp.resource")
    assert isinstance(HydrAMP(), AutoencoderResource)
    assert not issubclass(HydrAMP, Autoencoder)
    assert inspect.getsourcefile(Pipeline).endswith("core/wrapper/pipeline.py")


def test_default_retention_matches_the_mandatory_evaluation_contract():
    settings = StorageSettings()
    assert settings.save_lineage is True
    assert settings.metrics_level == "basic"
    assert settings.reconstruction == "none"
    assert set(settings.__dataclass_fields__).isdisjoint(
        {"save_terminal", "save_observation_history", "profile"}
    )
    assert set(ps.__all__) == {
        "filters",
        "flow",
        "mutations",
        "oracles",
        "saving",
        "walkers",
    }
