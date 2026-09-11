"""Command-line interface for PepCompass runtime execution."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from pep_compass.core.estimation import estimate_pipeline_stability
from pep_compass.core.validation import (
    diagnose_pipeline_specification,
    validate_pipeline_specification,
    validate_registered_components,
)
from pep_compass.runtime.backends import execute_subprocess_plan, write_slurm_array_script
from pep_compass.runtime.configuration import load_runtime_configuration
from pep_compass.runtime.configuration.pipeline import parse_pipeline_specification
from pep_compass.runtime.device import inspect_torch_runtime, validate_execution_device
from pep_compass.runtime.output import ResultWriter
from pep_compass.runtime.planning.plan import ExecutionPlan, materialize_execution_plan
from pep_compass.runtime.planning.validation import validate_execution_plan
from pep_compass.runtime.runner import RuntimeRunner
from pep_compass.runtime.test_run import TestRunPolicy, constrain_pipeline_for_test_run
from pep_compass.runtime.workflows import ComposableWorkflow


def _parser() -> argparse.ArgumentParser:
    """Build the PepCompass command-line parser."""
    parser = argparse.ArgumentParser(prog="pep-compass")
    commands = parser.add_subparsers(dest="command", required=True)

    # Production execution
    run = commands.add_parser("run", help="Execute a PepCompass configuration.")
    _add_configuration_arguments(run)
    run.add_argument("--backend", choices=("local", "subprocess", "slurm"))
    run.add_argument("--max-workers", type=int)
    run.add_argument("--resume", action="store_true", default=None)
    run.add_argument("--continue-on-error", action="store_true", default=None)
    run.add_argument("--slurm-script", type=Path)
    run.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)

    # Model-free graph validation and estimation
    dry_run = commands.add_parser(
        "dry-run",
        help="Validate and estimate a configuration without loading models.",
    )
    _add_configuration_arguments(dry_run)

    # Bounded execution with real models and disabled result persistence
    test_run = commands.add_parser(
        "test-run",
        help="Execute a bounded pipeline with real models and no result output.",
    )
    _add_configuration_arguments(test_run)
    test_run.add_argument("--tasks", type=int, default=2)
    test_run.add_argument("--iterations", type=int, default=1)
    test_run.add_argument("--max-candidates", type=int, default=2)
    test_run.add_argument("--max-oracle-calls", type=int, default=2)

    # REMARK: PoGS will be exposed as a separate workflow command.
    return parser


def _add_configuration_arguments(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by configuration-based commands."""
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--working-directory", type=Path, default=Path.cwd())
    parser.add_argument("--run-index", type=int, action="append")
    parser.add_argument(
        "--device",
        dest="autoencoder_device",
        help="Override autoencoder.device from the configuration (for example cpu or cuda).",
    )


def main() -> None:
    """Validate, test, or execute one runtime configuration."""
    arguments = _parser().parse_args()
    working_directory = arguments.working_directory.resolve()
    configuration_path = arguments.configuration.resolve()
    configuration = load_runtime_configuration(configuration_path)
    if arguments.autoencoder_device is not None:
        configuration = replace(
            configuration,
            autoencoder=replace(
                configuration.autoencoder,
                device=arguments.autoencoder_device,
            ),
        )

    if arguments.command == "dry-run":
        _run_dry_run(configuration, working_directory, arguments.run_index)
        return
    if arguments.command == "test-run":
        _run_test_run(configuration, working_directory, arguments)
        return
    if arguments.command == "run":
        _run_production(configuration, configuration_path, working_directory, arguments)
        return
    raise ValueError(f"Unsupported command: {arguments.command}")


def _run_dry_run(configuration, working_directory: Path, run_indices) -> None:
    """Validate every selected variant and print static graph estimates."""
    plan = _materialize_selected_plan(configuration, working_directory, run_indices)
    seen_variants: set[str] = set()
    for entry in plan.entries:
        if entry.variant.variant_id not in seen_variants:
            specification = parse_pipeline_specification(entry.variant.pipeline)
            validate_pipeline_specification(specification)
            validate_registered_components(specification)
            estimate = estimate_pipeline_stability(
                specification,
                input_candidates=1,
                latent_dimension=1,
            )
            diagnostics = diagnose_pipeline_specification(specification)
            _print_dry_run_report(entry.variant.variant_id, estimate, diagnostics)
            seen_variants.add(entry.variant.variant_id)
        print(
            f"{entry.index:05d} {entry.run_id} {entry.variant.variant_id} "
            f"{entry.task.task_id} seed={entry.seed}"
        )


def _run_test_run(configuration, working_directory: Path, arguments) -> None:
    """Execute a bounded local plan without constructing output writers."""
    policy = TestRunPolicy(
        tasks=arguments.tasks,
        iterations=arguments.iterations,
        candidates=arguments.max_candidates,
        oracle_calls=arguments.max_oracle_calls,
    )
    pipeline, overrides = constrain_pipeline_for_test_run(configuration.pipeline, policy)
    configuration = replace(
        configuration,
        pipeline=pipeline,
        experiment=replace(configuration.experiment, output_directory=None),
        execution=replace(
            configuration.execution,
            backend="local",
            max_workers=1,
            resume=False,
        ),
    )
    plan = _materialize_selected_plan(
        configuration,
        working_directory,
        arguments.run_index,
    )
    if arguments.run_index is None:
        plan = ExecutionPlan(plan.entries[: policy.tasks])
    print("test-run overrides:")
    for override in overrides:
        print(f"  {override}")
    _inspect_and_print_runtime(configuration.autoencoder.device)
    model_started_at = perf_counter()
    workflow = ComposableWorkflow(configuration.autoencoder)
    print(
        "model initialization: "
        f"duration={perf_counter() - model_started_at:.6f}s"
    )
    executions = RuntimeRunner(
        configuration,
        workflow,
        writer=None,
        capture_diagnostics=True,
    ).run(plan)
    for execution in executions:
        _print_test_run_report(execution)
    if any(execution.status == "failed" for execution in executions):
        raise SystemExit(1)


def _run_production(
    configuration,
    configuration_path: Path,
    working_directory: Path,
    arguments,
) -> None:
    """Dispatch a complete production execution through its selected backend."""
    execution = configuration.execution
    execution = replace(
        execution,
        backend=arguments.backend or execution.backend,
        max_workers=arguments.max_workers or execution.max_workers,
        resume=execution.resume if arguments.resume is None else arguments.resume,
        continue_on_error=(
            execution.continue_on_error
            if arguments.continue_on_error is None
            else arguments.continue_on_error
        ),
    )
    configuration = replace(configuration, execution=execution)
    plan = _materialize_selected_plan(
        configuration,
        working_directory,
        arguments.run_index,
    )

    if execution.backend == "subprocess" and not arguments.worker:
        execute_subprocess_plan(
            plan,
            configuration_path,
            working_directory=working_directory,
            max_workers=execution.max_workers,
            autoencoder_device=arguments.autoencoder_device,
        )
        return
    if execution.backend == "slurm" and not arguments.worker:
        if arguments.slurm_script is None:
            raise ValueError("--slurm-script is required for the Slurm backend.")
        write_slurm_array_script(
            plan,
            configuration_path,
            arguments.slurm_script,
            working_directory=working_directory,
            settings=execution.slurm,
            autoencoder_device=arguments.autoencoder_device,
        )
        return

    _inspect_and_print_runtime(configuration.autoencoder.device)
    output = configuration.experiment.output_directory
    writer = None
    if output is not None:
        if not output.is_absolute():
            output = working_directory / output
        writer = ResultWriter(output)
    workflow = ComposableWorkflow(configuration.autoencoder)
    executions = RuntimeRunner(configuration, workflow, writer).run(plan)
    if any(execution.status == "failed" for execution in executions):
        raise SystemExit(1)


def _materialize_selected_plan(
    configuration,
    working_directory: Path,
    run_indices,
) -> ExecutionPlan:
    """Materialize, select, and validate one deterministic execution plan."""
    plan = materialize_execution_plan(
        configuration,
        working_directory=working_directory,
    ).select(set(run_indices) if run_indices else None)
    validate_execution_plan(plan)
    return plan


def _print_dry_run_report(variant_id, estimate, diagnostics) -> None:
    """Print an explicit per-node cardinality and collision report."""
    print(f"variant {variant_id}")
    print("  candidate bounds:")
    for node in estimate.nodes:
        if node.operation == "flow":
            continue
        print(
            f"    {node.path} [{node.operation}] "
            f"input={_bound(node.input_candidates)} "
            f"output={_bound(node.output_candidates)} "
            f"peak={_bound(node.peak_candidates)}"
        )
        if node.uncertainty is not None:
            print(f"      unknown because: {node.uncertainty}")
    print(
        "  summary: "
        f"output={_bound(estimate.output_candidates_upper)} "
        f"peak={_bound(estimate.peak_candidates_upper)}"
    )
    print("  diagnostics:")
    if not diagnostics:
        print("    none")
    for diagnostic in diagnostics:
        print(
            f"    {diagnostic.severity} {diagnostic.code} "
            f"at {diagnostic.path}: {diagnostic.message}"
        )


def _print_test_run_report(execution) -> None:
    """Print timings, cardinalities, and memory observations for one test run."""
    print(
        f"{execution.entry.run_id} status={execution.status} "
        f"duration={execution.duration_seconds:.6f}s "
        f"candidates={execution.candidate_count} error={execution.error}"
    )
    print("  steps:")
    if not execution.step_records:
        print("    none")
    for record in execution.step_records:
        print(
            f"    {'/'.join(record.path)} input={record.input_size} "
            f"output={record.output_size} duration={record.duration_seconds:.6f}s "
            f"generated={record.generated_candidates_before}"
            f"->{record.generated_candidates_after} "
            f"oracle_calls={record.oracle_calls_before}->{record.oracle_calls_after}"
        )
    print("  memory:")
    for snapshot in execution.memory_snapshots:
        print(
            f"    {snapshot.label} candidates={snapshot.candidates} "
            f"rss={_bytes(snapshot.rss_bytes)} "
            f"cuda_allocated={_bytes(snapshot.cuda_allocated_bytes)} "
            f"cuda_reserved={_bytes(snapshot.cuda_reserved_bytes)} "
            f"cuda_peak={_bytes(snapshot.cuda_peak_bytes)} "
            f"batch={_bytes(snapshot.batch_bytes)} "
            f"detail_trigger={snapshot.detail_trigger or '-'}"
        )
    print("  final candidates:")
    if not execution.final_candidates:
        print("    none")
    elif all(score is None for _, score in execution.final_candidates):
        print("    score field: not produced (the executed pipeline has no oracle score)")
    for index, (sequence, score) in enumerate(execution.final_candidates):
        rendered_score = "N/A" if score is None else f"{score:.8g}"
        print(f"    {index}: sequence={sequence} score={rendered_score}")


def _inspect_and_print_runtime(device: str):
    """Print and validate the effective Torch and autoencoder device selection."""
    runtime = inspect_torch_runtime()
    compiled_cuda = runtime.compiled_cuda or "none (CPU-only build)"
    print(
        "runtime: "
        f"torch={runtime.version} compiled_cuda={compiled_cuda} "
        f"cuda_available={runtime.cuda_available} "
        f"cuda_devices={runtime.cuda_device_count} "
        f"autoencoder_device={device}"
    )
    if device == "cpu" and runtime.compiled_cuda is not None:
        print(
            "runtime note: CUDA Torch is installed, but autoencoder.device=cpu; "
            "CUDA is intentionally not used by the autoencoder."
        )
    try:
        validate_execution_device(device, runtime)
    except RuntimeError as error:
        raise SystemExit(f"pep-compass: error: {error}") from None
    return runtime


def _bound(value: int | None) -> str:
    """Render an explicitly unknown or finite candidate bound."""
    return "UNKNOWN" if value is None else str(value)


def _bytes(value: int | None) -> str:
    """Render bytes in a compact binary unit without hiding unavailable data."""
    if value is None:
        return "N/A"
    amount = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if amount < 1024 or unit == "TiB":
            return f"{amount:.2f}{unit}"
        amount /= 1024
    raise AssertionError("Unreachable byte unit conversion.")


if __name__ == "__main__":
    main()
