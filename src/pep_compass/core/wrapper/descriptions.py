"""Discover public exports and present constructor documentation as tables."""

from __future__ import annotations

import html
import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal

    import pandas as pd


def _text(value: object) -> str:
    """Text.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: str
    """
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return "None" if value is None else str(value)


def _prose(value: str) -> str:
    """Normalize inline reStructuredText without interpreting HTML.

    :param value: Value consumed by this conversion or transformation.
    :type value: str
    :return: Result of this operation.
    :rtype: str
    """
    value = re.sub(r":(?:class|func|meth|mod|attr|obj):`([^`]+)`", r"\1", value)
    return " ".join(value.replace("``", "").split())


def _doc_parts(value: str) -> tuple[str, dict[str, str]]:
    """Separate narrative and parameter descriptions from Sphinx fields.

    :param value: Raw factory documentation, optionally containing escaped newlines.
    :return: Narrative text and a mapping of parameter names to help text.

    :type value: str
    :rtype: tuple[str, dict[str, str]]
    """
    narrative, parameters = [], {}
    current = None
    in_fields = False
    for line in value.replace("\\n", "\n").splitlines():
        match = re.match(r"\s*:param\s+(?:[\w.\[\], |]+\s+)?(\w+)\s*:\s*(.*)", line)
        if match:
            current = match.group(1)
            parameters[current] = match.group(2)
            in_fields = True
        elif re.match(r"\s*:[\w]+(?:\s+[^:]+)?:", line):
            current = None
            in_fields = True
        elif current is not None:
            parameters[current] += " " + line.strip()
        elif not in_fields:
            narrative.append(line)
    return _prose(" ".join(narrative)), {
        key: _prose(text) for key, text in parameters.items()
    }


def _definition_rows(
    definition: Mapping[str, object],
    section: str = "",
    values: Mapping[str, object] | None = None,
    *,
    technical: bool = False,
) -> list[dict[str, object]]:
    """Flatten nested definitions without discarding internal-stage parameters.

    :param definition: Definition.
    :type definition: Mapping[str, object]
    :param section: Section.
    :type section: str
    :param values: Values consumed by this conversion or transformation.
    :type values: Mapping[str, object] | None
    :param technical: Whether to include internal resource and batch-field contracts.
    :type technical: bool
    :return: Result of this operation.
    :rtype: list[dict[str, object]]
    """
    rows = []
    prose, parameter_help = _doc_parts(definition.get("description", ""))
    fields = ["component", "strategy", "description"]
    if technical:
        fields.extend(("services", "input_fields", "output_fields", "optional_inputs"))
    for key in fields:
        if key in definition:
            rows.append(
                {
                    "section": section + "overview",
                    "name": key,
                    "value": prose if key == "description" else definition[key],
                }
            )
    for parameter in definition.get("parameters", ()):
        row = {"section": section + "parameters", **parameter}
        row["description"] = _prose(
            parameter.get("description", "")
        ) or parameter_help.get(parameter["name"], "")
        if not parameter.get("has_default", False):
            row["default"] = "<required>"
        row.pop("has_default", None)
        if values is not None and parameter["name"] in values:
            row["value"] = values[parameter["name"]]
        rows.append(row)
    for model in definition.get("model_descriptions", ()):
        rows.append(
            {
                "section": section + "models",
                "name": model["name"],
                "value": model if technical else model["name"],
            }
        )
    if not definition.get("model_descriptions") and definition.get("models"):
        rows.append(
            {
                "section": section + "models",
                "name": "available",
                "value": definition["models"],
            }
        )
    for slot, choices in definition.get("nested_strategies", {}).items():
        for choice in choices:
            rows.extend(
                _definition_rows(
                    choice,
                    f"{section}internal.{slot}.{choice['strategy']}.",
                    technical=technical,
                )
            )
    return rows


def description_frame(
    value: dict[str, object] | list[dict[str, object]], *, technical: bool = False
) -> pd.DataFrame:
    """Return one DataFrame for public choices or a complete detailed definition.

    :param value: Raw wrapper description records.
    :return: Catalog table or long-form detail table, retaining nested definitions.
    :rtype: pd.DataFrame

    :type value: dict[str, object] | list[dict[str, object]]
    :param technical: Whether to include internal resource and batch-field contracts.
    :type technical: bool
    """
    import pandas as pd

    if isinstance(value, list):
        return pd.DataFrame(
            [
                {
                    "component": item["component"],
                    "strategy": item["strategy"],
                    "description": _doc_parts(item.get("description", ""))[0],
                    "internal_stages": ", ".join(item.get("nested_strategies", {})),
                }
                for item in value
            ],
            columns=[
                "component",
                "strategy",
                "description",
                "internal_stages",
            ],
        )
    if "node" not in value:
        rows = _definition_rows(
            value, values=value.get("configuration"), technical=technical
        )
    else:
        node = value["node"]
        rows = [
            {"section": "placement", "name": name, "value": item}
            for name, item in {
                "node_id": node["node_id"],
                "name": node.get("name"),
                "position": value["position"],
                "parent_id": value["parent_id"],
                "previous_id": value["previous_id"],
                "next_id": value["next_id"],
            }.items()
        ]
        if "definition" in value:
            rows.extend(
                _definition_rows(
                    value["definition"],
                    values=value["resolved_parameters"],
                    technical=technical,
                )
            )
        else:
            rows.append(
                {"section": "configuration", "name": "declaration", "value": node}
            )
        for name, fields in (
            value.get("resolved_contract", {}) if technical else {}
        ).items():
            rows.append({"section": "resolved_contract", "name": name, "value": fields})
    return pd.DataFrame(rows, dtype=object).reindex(
        columns=[
            "section",
            "name",
            "type",
            "required",
            "default",
            "value",
            "description",
        ]
    )


def _notebook_available() -> bool:
    """Notebook available.

    :return: Result of this operation.
    :rtype: bool
    """
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    return getattr(get_ipython(), "kernel", None) is not None


def _readable(frame: pd.DataFrame) -> pd.DataFrame:
    """Format containers for presentation without mutating the returned data table.

    :param frame: Frame.
    :type frame: pd.DataFrame
    :return: Result of this operation.
    :rtype: pd.DataFrame
    """
    return frame.map(
        lambda value: (
            _text(value)
            if value is None or isinstance(value, (Mapping, list, tuple))
            else value
        )
    ).fillna("")


def _html(frame: pd.DataFrame) -> str:
    """Render detail sections with collapsed internal strategies and escaped text.

    :param frame: Frame.
    :type frame: pd.DataFrame
    :return: Result of this operation.
    :rtype: str
    """
    parts = [
        '<div class="pep-description"><style>.pep-description table{border-collapse:collapse;text-align:left;font-size:13px}.pep-description td,.pep-description th{padding:6px 10px;border-bottom:1px solid #dde3ec;white-space:pre-wrap;max-width:560px;vertical-align:top}.pep-description summary{cursor:pointer;font-weight:600;margin:12px 0}.pep-description{overflow:auto}</style>'
    ]
    if "section" not in frame:
        parts.append(frame.to_html(index=False, border=0, escape=True))
    else:
        for section, group in frame.groupby("section", sort=False):
            title = section.replace("_", " ").replace(".", " / ").capitalize()
            opened = "" if section.startswith("internal.") else " open"
            table = group.drop(columns="section")
            table = table.loc[:, (table != "").any(axis=0)]
            parts.extend(
                [
                    f"<details{opened}><summary>{html.escape(title)}</summary>",
                    table.to_html(index=False, border=0, escape=True),
                    "</details>",
                ]
            )
    parts.append("</div>")
    return "".join(parts)


def present_description(
    value: dict[str, object] | list[dict[str, object]],
    mode: Literal["raw", "pd", "display"] = "display",
    *,
    technical: bool = False,
) -> dict[str, object] | list[dict[str, object]] | pd.DataFrame | None:
    """Return raw records/a DataFrame or display readable tables automatically.

    :param value: Raw wrapper description; never changed by presentation.
    :param mode: ``raw``, ``pd`` or ``display`` (default).
    :return: Unchanged records, a DataFrame, or None after displaying tables.
    :raises ValueError: If the presentation mode is unsupported.

    :type value: dict[str, object] | list[dict[str, object]]
    :type mode: Literal['raw', 'pd', 'display']
    :param technical: Whether to include internal resource and batch-field contracts.
    :type technical: bool
    :rtype: dict[str, object] | list[dict[str, object]] | pd.DataFrame | None
    """
    if mode not in {"raw", "pd", "display"}:
        raise ValueError("display must be 'raw', 'pd' or 'display'.")
    if mode == "raw":
        return value
    frame = description_frame(value, technical=technical)
    if mode == "pd":
        return frame
    readable = _readable(frame)
    if _notebook_available():
        from IPython.display import HTML, display

        display(HTML(_html(readable)))
    elif "section" in readable:
        for section, group in readable.groupby("section", sort=False):
            print(section.replace("_", " ").replace(".", " / ").capitalize())
            table = group.drop(columns="section")
            print(table.loc[:, (table != "").any(axis=0)].to_string(index=False))
    else:
        print(readable.to_string(index=False))
    return None


def definition(value: object) -> dict[str, object]:
    """Describe the real constructor and its resolved values without loading weights.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: dict[str, object]
    """
    import inspect

    from pep_compass.steps.base import Step
    from pep_compass.utils.configuration import Configurable, plain

    target = (
        value if inspect.isclass(value) or inspect.isfunction(value) else type(value)
    )
    prose, help_text = _doc_parts(inspect.getdoc(target) or "")
    if inspect.isclass(target):
        for parent in reversed(target.__mro__[1:]):
            _, inherited_help = _doc_parts(inspect.getdoc(parent) or "")
            help_text = {**inherited_help, **help_text}
        _, constructor_help = _doc_parts(inspect.getdoc(target.__init__) or "")
        help_text.update(constructor_help)
    parameters = []
    for name, parameter in inspect.signature(target).parameters.items():
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        has_default = parameter.default is not inspect.Parameter.empty
        parameters.append(
            {
                "name": name,
                "type": str(parameter.annotation)
                if parameter.annotation is not inspect.Parameter.empty
                else "",
                "required": not has_default,
                "has_default": has_default,
                "default": plain(parameter.default) if has_default else None,
                "description": help_text.get(name, ""),
            }
        )
    result = {
        "component": getattr(
            target,
            "category",
            target.__module__.split(".")[-2] if "." in target.__module__ else "custom",
        ),
        "strategy": target.__name__,
        "description": prose,
        "parameters": parameters,
    }
    result["nested_strategies"] = {
        name: [] for name in getattr(target, "stage_parameters", ())
    }
    if isinstance(value, Configurable):
        result["configuration"] = value.configuration
        result["nested_strategies"] = {
            key: [definition(child)]
            for key, child in vars(value).items()
            if isinstance(child, Configurable) and not isinstance(child, Step)
        }
    if isinstance(value, Step):
        contract = value.contract()
        result.update(
            services=list(contract.resources),
            input_fields=list(contract.required_fields),
            output_fields=list(contract.produced_fields),
            optional_inputs=list(contract.optional_fields),
        )
    return result


def available(category: str | None = None) -> list[dict[str, object]]:
    """Read __all__ from public namespaces; no mutable catalogs or plugin scans.

    :param category: Optional public component category used to restrict the listing.
    :type category: str | None
    :return: Result of this operation.
    :rtype: list[dict[str, object]]
    """
    from importlib import import_module

    names = (
        (category,)
        if category
        else (
            "filters",
            "flow",
            "mutations",
            "oracles",
            "saving",
            "walkers",
            "autoencoder",
        )
    )
    records = []
    for name in names:
        module = import_module(
            "pep_compass." + (name if name == "autoencoder" else "steps." + name)
        )
        for exported in module.__all__:
            target = getattr(module, exported)
            if getattr(target, "_is_protocol", False):
                continue
            item = definition(target)
            item["component"] = name
            records.append(item)
    return records
