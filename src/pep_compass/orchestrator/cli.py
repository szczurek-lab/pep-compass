"""Hydra-native commands and isolated workers for wrapper-based experiments."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from pep_compass.orchestrator.configuration import load_plugins

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from omegaconf import DictConfig


def main(argv: Sequence[str] | None = None) -> None:
    """Execute the CLI; Hydra owns config overrides and multirun.

    :param argv: Optional explicit command arguments, otherwise sys.argv.
    :raises SystemExit: For invalid arguments or a failed worker.

    :type argv: Sequence[str] | None
    :rtype: None
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(
            "pep-compass {run|plan|worker} [arguments]\n"
            "  run/plan: Hydra defaults, key=value, --config-path, --config-name, --multirun\n"
            "  --plugin MODULE: import explicit application modules\n"
            "  worker PLAN --index N [--skip-completed]: execute one frozen job"
        )
        return
    command = arguments.pop(0)
    plugins = []
    while "--plugin" in arguments:
        index = arguments.index("--plugin")
        if index + 1 >= len(arguments):
            raise SystemExit("--plugin requires a Python module name.")
        plugins.append(arguments[index + 1])
        del arguments[index : index + 2]
    load_plugins(plugins)
    if command == "worker":
        from pep_compass.orchestrator.planning import load_plan
        from pep_compass.orchestrator.runner import Orchestrator

        parser = argparse.ArgumentParser(prog="pep-compass worker")
        parser.add_argument("plan", type=Path)
        parser.add_argument("--index", type=int, required=True)
        parser.add_argument("--skip-completed", action="store_true", default=None)
        selected = parser.parse_args(arguments)
        plan = load_plan(selected.plan)
        if not 0 <= selected.index < len(plan.entries):
            parser.error("index is outside this frozen plan")
        record = Orchestrator().run_entry(
            plan, plan.entries[selected.index], skip_completed=selected.skip_completed
        )
        print("PEP_COMPASS_RESULT=" + json.dumps(asdict(record)))
        if record.status == "failed":
            raise SystemExit(1)
        return
    if command not in {"run", "plan"}:
        raise SystemExit(f"Unknown command: {command!r}; use --help.")
    previous = sys.argv
    try:
        sys.argv = [previous[0], *arguments]
        _hydra_entry(command, plugins)()
    finally:
        sys.argv = previous


def _hydra_entry(command: str, plugins: Sequence[str]) -> Callable[[DictConfig], None]:
    """Hydra entry.

    :param command: Command.
    :type command: str
    :param plugins: Plugins.
    :type plugins: Sequence[str]
    :return: Result of this operation.
    :rtype: Callable[[DictConfig], None]
    """
    import hydra
    from hydra.utils import get_original_cwd
    from omegaconf import open_dict

    from pep_compass.orchestrator.planning import plan_configuration
    from pep_compass.orchestrator.runner import Orchestrator

    @hydra.main(version_base="1.3", config_path="conf", config_name="config")
    def execute(configuration: DictConfig) -> None:
        """Execute.

        :param configuration: Resolved configuration for this operation.
        :type configuration: DictConfig
        :rtype: None
        """
        with open_dict(configuration):
            configuration.plugins = list(
                dict.fromkeys([*configuration.get("plugins", []), *plugins])
            )
        plan = plan_configuration(configuration, working_directory=get_original_cwd())
        if command == "plan":
            print(json.dumps(plan.to_dict(), indent=2))
            return
        records = Orchestrator().run(plan)
        if isinstance(records, Path):
            print("Slurm script:", records)
        else:
            print(json.dumps([asdict(record) for record in records], indent=2))
            if any(record.status == "failed" for record in records):
                raise RuntimeError(
                    "One or more orchestrated jobs failed; see status records."
                )

    return execute


if __name__ == "__main__":
    main()
