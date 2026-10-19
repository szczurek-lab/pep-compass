"""Dependency-free graph views over declarations, never over running models."""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

from pep_compass.utils.logger import get_custom_logger

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Self

    from pep_compass.data_schemas.graph import NodeSpec, PipelineSchema


logger = get_custom_logger(__name__)


class GraphView(str):
    """A normal string with optional inline SVG display in a notebook."""

    def __new__(cls, value: str, format: str) -> Self:
        """New.

        :param value: Value consumed by this conversion or transformation.
        :type value: str
        :param format: Format.
        :type format: str
        :return: Result of this operation.
        :rtype: Self
        """
        instance = super().__new__(cls, value)
        instance.format = format
        return instance

    def _repr_mimebundle_(self, **kwargs: object) -> dict[str, str]:
        """Repr mimebundle.

        :param kwargs: Additional named arguments forwarded to the underlying interface.
        :type kwargs: object
        :return: Result of this operation.
        :rtype: dict[str, str]
        """
        return (
            {"image/svg+xml": str(self), "text/plain": "PepCompass computation graph"}
            if self.format == "svg"
            else {"text/plain": str(self)}
        )


def _information(node: NodeSpec) -> tuple[str, dict[str, object]]:
    """Information.

    :param node: Node.
    :type node: NodeSpec
    :return: Result of this operation.
    :rtype: tuple[str, dict[str, object]]
    """
    component = node.instantiate()
    return f"{node.category}: {node.label}", component.display_info()


def _compact(value: object, limit: int = 100) -> object:
    """Compact.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :param limit: Limit.
    :type limit: int
    :return: Result of this operation.
    :rtype: object
    """
    text = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
    return text if len(text) <= limit else text[: limit - 1] + "…"


def render(schema: PipelineSchema, *, format: str) -> PipelineSchema | str:
    """Render identities, selected parameters, flow and resource relationships.

    :param schema: Immutable PipelineSchema snapshot.
    :param format: text, mermaid or svg.
    :return: A portable string; no optional system executable is required.

    :type schema: PipelineSchema
    :type format: str
    :rtype: PipelineSchema | str
    """
    if format == "object":
        return schema
    if format == "json":
        return json.dumps(schema.to_dict(), indent=2, allow_nan=False)
    logger.debug(
        "Rendering pipeline format=%s root_nodes=%s.", format, len(schema.steps)
    )
    if format == "text":
        lines = []
        if schema.autoencoder is not None:
            lines.append(
                f"autoencoder: {schema.autoencoder['_target_'].rsplit('.', 1)[-1]}/{schema.autoencoder.get('model', '')} [shared resource]"
            )
        for node, parent, position in schema.walk():
            kind, parameters = _information(node)
            lines.append(
                f"{position} [{node.node_id}] {node.name or kind}: {_compact(parameters, 200)} parent={parent or 'root'}"
            )
        return "\n".join(lines) or "Empty computation graph"
    if format == "mermaid":
        return _mermaid(schema)
    if format == "svg":
        return _svg(schema)
    raise ValueError("format must be object, json, text, mermaid or svg.")


def _mermaid(schema: PipelineSchema) -> str:
    """Mermaid.

    :param schema: Immutable declaration or its explicitly supported serialized representation.
    :type schema: PipelineSchema
    :return: Result of this operation.
    :rtype: str
    """
    identities = {
        node.node_id: f"n{index}" for index, (node, _, _) in enumerate(schema.walk())
    }
    positions = {node.node_id: position for node, _, position in schema.walk()}
    lines = ["flowchart TD"]
    if schema.autoencoder is not None:
        encoder = html.escape(schema.autoencoder["_target_"].rsplit(".", 1)[-1])
        model = html.escape(schema.autoencoder.get("model", ""))
        lines.append(f'  autoencoder["Autoencoder: {encoder} / {model}"]')

    def label(node: NodeSpec) -> str:
        """Label.

        :param node: Node.
        :type node: NodeSpec
        :return: Result of this operation.
        :rtype: str
        """
        kind, parameters = _information(node)
        text = f"{positions[node.node_id]} · {node.name or _caption(node)} | {_caption(node)} | {node.node_id} | {_compact(parameters)}"
        return html.escape(text, quote=True).replace("\n", " ")

    def sequence(nodes: Sequence[NodeSpec]) -> tuple[str | None, str | None]:
        """Sequence.

        :param nodes: Nodes.
        :type nodes: Sequence[NodeSpec]
        :return: Result of this operation.
        :rtype: tuple[str | None, str | None]
        """
        first = previous = None
        for node in nodes:
            entry, exit = visit(node)
            if previous is not None:
                lines.append(f"  {previous} --> {entry}")
            first = first or entry
            previous = exit
        return first, previous

    def visit(node: NodeSpec) -> tuple[str, str]:
        """Visit.

        :param node: Node.
        :type node: NodeSpec
        :return: Result of this operation.
        :rtype: tuple[str, str]
        """
        identity = identities[node.node_id]
        if not node.children:
            lines.append(f'  {identity}["{label(node)}"]')
            if (
                schema.autoencoder is not None
                and "autoencoder" in node.instantiate().contract().resources
            ):
                lines.append(f"  autoencoder -. resource .-> {identity}")
            return identity, identity
        lines.append(
            f'  subgraph g_{identity}["{html.escape(node.name or _caption(node))}"]'
        )
        lines.append(f'  {identity}["{label(node)}"]')
        end = identity + "_out"
        lines.append(f'  {end}(("Return to parent"))')
        if node.kind in {"loop", "sequence"}:
            first, last = sequence(node.children["steps"])
            if first is not None:
                lines.extend(
                    (
                        f"  {identity} --> {first}",
                        f"  {last} -. next iteration .-> {first}",
                        f"  {last} --> {end}",
                    )
                )
            else:
                lines.append(f"  {identity} --> {end}")
        elif node.kind == "parallel":
            for name, branch in node.children["branches"].items():
                first, last = sequence(branch)
                if first is not None:
                    lines.extend(
                        (
                            f'  {identity} -->|"{html.escape(name)}"| {first}',
                            f"  {last} --> {end}",
                        )
                    )
                else:
                    lines.append(f"  {identity} --> {end}")
        else:
            walker, _ = visit(node.children["walker"])
            generator, _ = visit(node.children["mutation_generator"])
            first, last = sequence(node.children["filters"])
            lines.extend(
                (
                    f"  {identity} --> {walker}",
                    f"  {walker} -. next point .-> {walker}",
                    f"  {walker} --> {generator}",
                )
            )
            lines.append(f"  {generator} --> {first or end}")
            if last is not None:
                lines.append(f"  {last} --> {end}")
            if node.parameters.get("include_walk_points", True):
                lines.append(f"  {walker} --> {end}")
        lines.append("  end")
        return identity, end

    sequence(schema.steps)
    return "\n".join(lines)


def _caption(node: NodeSpec) -> str:
    """Caption.

    :param node: Node.
    :type node: NodeSpec
    :return: Result of this operation.
    :rtype: str
    """
    return f"{node.category.replace('_', ' ').capitalize()} · {node.label}"


def _svg(schema: PipelineSchema) -> str:
    """Route nested entry right, return left, and feedback to its actual target.

    :param schema: Immutable declaration or its explicitly supported serialized representation.
    :type schema: PipelineSchema
    :return: Result of this operation.
    :rtype: str
    """
    from textwrap import shorten, wrap

    positions = {node.node_id: position for node, _, position in schema.walk()}
    shapes, edges = [], []

    # Measure the owned subgraphs before assigning coordinates.
    def dimensions(node: NodeSpec) -> tuple[float, float]:
        """Dimensions.

        :param node: Node.
        :type node: NodeSpec
        :return: Result of this operation.
        :rtype: tuple[float, float]
        """
        if not node.children:
            return 520, 126
        if node.kind in {"loop", "sequence"}:
            width, height = sequence_size(node.children["steps"])
        elif node.kind == "parallel":
            sizes = [
                sequence_size(branch) for branch in node.children["branches"].values()
            ]
            width = sum(size[0] for size in sizes) + 40 * max(0, len(sizes) - 1)
            height = max((size[1] for size in sizes), default=0)
        else:
            width, height = sequence_size(
                (
                    node.children["walker"],
                    node.children["mutation_generator"],
                    *node.children["filters"],
                )
            )
        return width + 96, height + 178

    def sequence_size(nodes: Sequence[NodeSpec]) -> tuple[float, float]:
        """Sequence size.

        :param nodes: Nodes.
        :type nodes: Sequence[NodeSpec]
        :return: Result of this operation.
        :rtype: tuple[float, float]
        """
        sizes = [dimensions(node) for node in nodes]
        return max((size[0] for size in sizes), default=520), sum(
            size[1] for size in sizes
        ) + 32 * max(0, len(sizes) - 1)

    def path(
        points: Sequence[tuple[float, float]],
        *,
        kind: str = "flow",
        owner: str = "",
        dashed: bool = False,
        arrow: bool = True,
        title: str = "",
    ) -> None:
        """Path.

        :param points: Points.
        :type points: Sequence[tuple[float, float]]
        :param kind: Kind of recorded artifact or exception type, as declared by this operation.
        :type kind: str
        :param owner: Owner.
        :type owner: str
        :param dashed: Dashed.
        :type dashed: bool
        :param arrow: Arrow.
        :type arrow: bool
        :param title: Title.
        :type title: str
        :rtype: None
        """
        coordinates = " L".join(f"{x},{y}" for x, y in points)
        dash = ' stroke-dasharray="6 4"' if dashed else ""
        marker = ' marker-end="url(#arrow)"' if arrow else ""
        edges.append(
            f'<path data-kind="{kind}" data-owner="{html.escape(owner, quote=True)}" d="M{coordinates}" stroke="#52647d" stroke-width="1.4" fill="none"{marker}{dash}><title>{html.escape(title)}</title></path>'
        )

    def text(
        x: float, y: float, value: str, size: float = 12, bold: bool = False
    ) -> None:
        """Text.

        :param x: X.
        :type x: float
        :param y: Y.
        :type y: float
        :param value: Value consumed by this conversion or transformation.
        :type value: str
        :param size: Size.
        :type size: float
        :param bold: Bold.
        :type bold: bool
        :rtype: None
        """
        weight = ' font-weight="600"' if bold else ""
        shapes.append(
            f'<text x="{x}" y="{y}" font-size="{size}"{weight}>{html.escape(str(value))}</text>'
        )

    def sequence(
        nodes: Sequence[NodeSpec], x: float, y: float
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Sequence.

        :param nodes: Nodes.
        :type nodes: Sequence[NodeSpec]
        :param x: X.
        :type x: float
        :param y: Y.
        :type y: float
        :return: Result of this operation.
        :rtype: tuple[tuple[float, float] | None, tuple[float, float] | None]
        """
        first = previous = None
        for node in nodes:
            _, height = dimensions(node)
            entry, exit = visit(node, x, y)
            if previous is not None:
                mid_y = entry[1] - 16
                path([previous, (previous[0], mid_y), (entry[0], mid_y), entry])
            first = first or entry
            previous = exit
            y += height + 32
        return first, previous

    # Child input/output ports expose the direction of changing hierarchy.
    def connect_body(
        node: NodeSpec,
        first_x: float,
        first_y: float,
        last: tuple[float, float],
        width: float,
        height: float,
        x: float,
        y: float,
    ) -> None:
        """Connect body.

        :param node: Node.
        :type node: NodeSpec
        :param first_x: First x.
        :type first_x: float
        :param first_y: First y.
        :type first_y: float
        :param last: Last.
        :type last: tuple[float, float]
        :param width: Width.
        :type width: float
        :param height: Height.
        :type height: float
        :param x: X.
        :type x: float
        :param y: Y.
        :type y: float
        :rtype: None
        """
        target = (first_x, first_y + 24)
        path(
            [
                (x + 18, y + 90),
                (first_x - 18, y + 90),
                (first_x - 18, target[1]),
                target,
            ],
            kind="enter",
            owner=node.node_id,
            title="Enter nested computation",
        )
        return_y = y + height - 44
        path(
            [last, (last[0], return_y), (x + 18, return_y)],
            kind="return",
            owner=node.node_id,
            title="Return candidates to the parent computation",
        )
        path(
            [
                (x + 18, return_y),
                (x + 18, y + height - 6),
                (x + width / 2, y + height - 6),
                (x + width / 2, y + height),
            ],
            kind="output",
            owner=node.node_id,
            arrow=False,
        )

    def visit(
        node: NodeSpec, x: float, y: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Visit.

        :param node: Node.
        :type node: NodeSpec
        :param x: X.
        :type x: float
        :param y: Y.
        :type y: float
        :return: Result of this operation.
        :rtype: tuple[tuple[float, float], tuple[float, float]]
        """
        width, height = dimensions(node)
        _, parameters = _information(node)
        leaf = not node.children
        caption = _caption(node)
        position = positions[node.node_id]
        compact_position = (
            position.replace(".steps.", ".")
            .replace(".mutation_generator.", ".mutations.")
            .replace(".branches.", ".")
        )
        title = shorten(
            f"{compact_position} · {node.name or caption}", width=64, placeholder=" …"
        )
        tooltip = f"{caption}\nPosition: {position}\n{node.node_id}\n" + json.dumps(
            parameters, ensure_ascii=False, default=str, indent=2
        )
        fill = "#edf4ff" if leaf else "#f6f8fc"
        shapes.append(
            f'<g data-node-id="{html.escape(node.node_id, quote=True)}"><title>{html.escape(tooltip)}</title><rect x="{x}" y="{y}" width="{width}" height="{height}" rx="10" fill="{fill}" stroke="#7386a4"/></g>'
        )
        text(x + 12, y + 23, title, 14, True)
        text(x + 12, y + 43, caption, 12)
        text(x + 12, y + 62, node.node_id, 10)
        parameter_text = " · ".join(
            f"{key.replace('_', ' ')} = {_text_value(value)}"
            for key, value in parameters.items()
        )
        for index, row in enumerate(
            wrap(parameter_text, width=74, max_lines=2, placeholder=" …")
        ):
            text(x + 12, y + 81 + index * 16, row, 11)
        if not leaf:
            body_y = y + 106
            if node.kind == "parallel":
                branch_x = x + 64
                for name, branch in node.children["branches"].items():
                    text(branch_x, body_y - 8, f"Branch: {name}", 11)
                    first, last = sequence(branch, branch_x, body_y)
                    if first is not None:
                        connect_body(node, branch_x, body_y, last, width, height, x, y)
                    branch_x += sequence_size(branch)[0] + 40
                note = "Merge branch outputs and return to the parent flow"
            else:
                nodes = (
                    node.children["steps"]
                    if node.kind in {"loop", "sequence"}
                    else (
                        node.children["walker"],
                        node.children["mutation_generator"],
                        *node.children["filters"],
                    )
                )
                first, last = sequence(nodes, x + 64, body_y)
                if first is not None:
                    connect_body(node, x + 64, body_y, last, width, height, x, y)
                    first_width, _ = dimensions(nodes[0])
                    target = (x + 64 + first_width, body_y + 24)
                    start = (
                        (last[0], last[1] + 10)
                        if node.kind in {"loop", "sequence"}
                        else (x + 64 + first_width, body_y + 88)
                    )
                    path(
                        [
                            start,
                            (x + width - 14, start[1]),
                            (x + width - 14, target[1]),
                            target,
                        ],
                        kind="feedback",
                        owner=node.node_id,
                        dashed=True,
                        title="Next loop iteration"
                        if node.kind in {"loop", "sequence"}
                        else "Next private walker point; mutations do not feed back",
                    )
                    if not node.kind in {"loop", "sequence"} and node.parameters.get(
                        "include_walk_points", True
                    ):
                        walker_height = dimensions(node.children["walker"])[1]
                        walker_output = (x + 64, body_y + walker_height - 12)
                        path(
                            [
                                walker_output,
                                (x + 38, walker_output[1]),
                                (x + 38, y + height - 44),
                                (x + 18, y + height - 44),
                            ],
                            kind="walk-output",
                            owner=node.node_id,
                            title="Include walker points in the output pool",
                        )
                note = (
                    "Repeat the body; return its final output"
                    if node.kind in {"loop", "sequence"}
                    else "Private walker feedback; mutations and filters fill the output pool"
                )
            text(x + 32, y + height - 18, note, 11)
        return (x + width / 2, y), (x + width / 2, y + height)

    width, height = sequence_size(schema.steps)
    top = 110 if schema.autoencoder is not None else 16
    first, _ = sequence(schema.steps, 20, top)
    if schema.autoencoder is not None:
        shapes.insert(
            0,
            f'<rect x="20" y="10" width="{width}" height="50" rx="8" fill="#e6f5eb" stroke="#4b8661"/>',
        )
        text(
            32,
            40,
            f"Shared autoencoder: {schema.autoencoder['_target_'].rsplit('.', 1)[-1]} / {schema.autoencoder.get('model', '')}",
            14,
            True,
        )
        if first:
            path(
                [(20 + width / 2, 60), first],
                kind="resource",
                dashed=True,
                title="Shared resource, not a computation step",
            )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width + 40}" height="{height + top + 20}" viewBox="0 0 {width + 40} {height + top + 20}"><defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0,0 L8,4 L0,8" fill="#52647d"/></marker></defs><g font-family="sans-serif" fill="#23334a">'
        + "".join(shapes + edges)
        + "</g></svg>"
    )


def _text_value(value: object) -> str:
    """Keep compact labels readable; exact parameter values remain in tooltips.

    :param value: Value consumed by this conversion or transformation.
    :type value: object
    :return: Result of this operation.
    :rtype: str
    """
    if isinstance(value, dict):
        if "method" in value:
            parameters = value.get("parameters", {})
            details = _text_value(parameters)
            return (
                f"{value['method']}({details})" if parameters else str(value["method"])
            )
        return ", ".join(
            f"{str(key).replace('_', ' ')}: {_text_value(item)}"
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return ", ".join(_text_value(item) for item in value)
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def graph_view(schema: PipelineSchema, format: str = "svg") -> GraphView:
    """Create the shared notebook representation for declared or compiled graphs.

    :param schema: Immutable declaration or its explicitly supported serialized representation.
    :type schema: PipelineSchema
    :param format: Format.
    :type format: str
    :return: Result of this operation.
    :rtype: GraphView
    """
    return GraphView(render(schema, format=format), format)
