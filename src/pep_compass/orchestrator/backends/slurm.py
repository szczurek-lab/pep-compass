"""Generate Slurm arrays using built-in or server-owned shell templates."""

from __future__ import annotations

from pathlib import Path
from shlex import quote
from string import Template
from typing import TYPE_CHECKING

from pep_compass.orchestrator.backends.base import ExecutionBackend
from pep_compass.orchestrator.backends.subprocess import worker_command
from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pep_compass.orchestrator.planning import ExecutionPlan
    from pep_compass.orchestrator.runner import Orchestrator


logger = get_custom_logger(__name__)
DEFAULT_TEMPLATE_PATH = (
    Path(__file__).resolve().parents[1] / "templates" / "slurm_array.sh"
)


class SlurmBackend(ExecutionBackend):
    """Generate a portable array script; submission remains an explicit user action.

    :param template_path: Optional server-owned template; resolved when planning.
    :param template: Inline template, frozen in the plan for workers.
    :param settings: Slurm directives: partition, account, time, memory,
        cpus_per_task and gpus. Complex site setup belongs in the template.
    """

    def __init__(
        self,
        template_path: str | Path | None = None,
        template: str | Path | None = None,
        settings: Mapping[str, str | int | float] | None = None,
    ) -> None:
        """Initialize this object with the declared configuration.

        :param template_path: Template path.
        :type template_path: str | Path | None
        :param template: Template.
        :type template: str | Path | None
        :param settings: Validated settings or a resolved settings manifest.
        :type settings: Mapping[str, str | int | float] | None
        :rtype: None
        """
        if template_path is not None and template is not None:
            raise ValueError("Select template_path or inline template, not both.")
        self.template_path, self.template = template_path, template
        self.settings = dict(settings or {})
        flags = {"partition", "account", "time", "memory", "cpus_per_task", "gpus"}
        if self.settings.keys() - flags:
            raise ValueError(
                "Unknown Slurm settings; use a site template for custom directives."
            )
        for key, value in self.settings.items():
            if (
                not str(value)
                or any(char.isspace() for char in str(value))
                or str(value).startswith("-")
            ):
                raise ValueError(f"Invalid Slurm directive value: {key}.")
            if key in {"cpus_per_task", "gpus"} and (
                isinstance(value, bool) or not isinstance(value, int) or value < 1
            ):
                raise ValueError(f"Slurm {key} must be a positive integer.")

    @classmethod
    def resolve_parameters(
        cls, parameters: Mapping[str, object], working_directory: str | Path
    ) -> dict[str, object]:
        """Resolve parameters.

        :param parameters: Named configuration values consumed by this operation.
        :type parameters: Mapping[str, object]
        :param working_directory: Working directory.
        :type working_directory: str | Path
        :return: Result of this operation.
        :rtype: dict[str, object]
        """
        values = dict(parameters)
        path = values.get("template_path")
        if path is not None and values.get("template") is not None:
            raise ValueError("Select template_path or inline template, not both.")
        if values.get("template") is None:
            source = DEFAULT_TEMPLATE_PATH if path is None else Path(path)
            if not source.is_absolute():
                source = Path(working_directory) / source
            values["template"] = source.read_text(encoding="utf-8")
            values["template_path"] = None
        return values

    def run(
        self, orchestrator: Orchestrator, plan: ExecutionPlan, plan_path: Path
    ) -> Path:
        """Run.

        :param orchestrator: Orchestrator.
        :type orchestrator: Orchestrator
        :param plan: Plan.
        :type plan: ExecutionPlan
        :param plan_path: Plan path.
        :type plan_path: Path
        :return: Result of this operation.
        :rtype: Path
        """
        if orchestrator.autoencoder is not None:
            raise ValueError(
                "Slurm jobs require serializable resource declarations with direct import targets."
            )
        template = (
            self.template
            if self.template is not None
            else DEFAULT_TEMPLATE_PATH.read_text(encoding="utf-8")
        )
        if self.template_path is not None:
            raise ValueError("Resolve the Slurm template while planning.")
        if "${command}" not in template or "${array_max}" not in template:
            raise ValueError(
                "Slurm templates require ${command} and ${array_max} placeholders."
            )
        command = worker_command(
            plan_path, 0, skip_completed=plan.options.skip_completed
        )
        command[command.index("--index") + 1] = '"${SLURM_ARRAY_TASK_ID:?}"'
        rendered_command = " ".join(
            item if item == '"${SLURM_ARRAY_TASK_ID:?}"' else quote(item)
            for item in command
        )
        flags = {"memory": "mem", "cpus_per_task": "cpus-per-task"}
        directives = "\n".join(
            f"#SBATCH --{flags.get(key, key)}={value}"
            for key, value in self.settings.items()
        )
        script = Template(template).safe_substitute(
            command=rendered_command,
            array_max=len(plan.entries) - 1,
            working_directory=quote(plan.working_directory),
            plan_path=quote(str(plan_path)),
            directives=directives,
        )
        target = Path(plan_path).with_suffix(".slurm.sh")
        target.write_text(script, encoding="utf-8")
        logger.info(
            "Generated Slurm array jobs=%s script=%s.", len(plan.entries), target
        )
        return target
