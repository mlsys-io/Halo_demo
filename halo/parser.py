from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import yaml

from .models import (
    DBQuery,
    Edge,
    GraphSpec,
    GraphValidationError,
    Node,
    QueryPlanOption,
)
from .utils import as_bool


class GraphTemplateParser:
    """Parses a YAML template into strongly typed graph objects."""

    def __init__(self, template_path: Path):
        self.template_path = Path(template_path)

    def parse(self) -> GraphSpec:
        data = yaml.safe_load(self.template_path.read_text())
        if not isinstance(data, dict) or "graph" not in data:
            raise GraphValidationError("Expected top-level 'graph' entry in template.")
        graph_data = data["graph"]
        name = graph_data.get("name", "unnamed")
        description = graph_data.get("description", "")
        nodes = self._parse_nodes(graph_data.get("nodes", []))
        edges = self._parse_edges(graph_data.get("edges", []))
        nodes, edges = self._expand_embedded_calls(nodes, edges)
        return GraphSpec(name=name, description=description, nodes=nodes, edges=edges)

    def _parse_nodes(self, nodes: Iterable[Dict[str, Any]]) -> Dict[str, Node]:
        parsed: Dict[str, Node] = {}
        for node in nodes:
            node_id = node.get("id")
            if not node_id:
                raise GraphValidationError("Every node must have an 'id'.")
            if node_id in parsed:
                raise GraphValidationError(f"Duplicate node id '{node_id}'.")
            db_queries = tuple(
                DBQuery(
                    name=query.get("name", "unnamed"),
                    sql=query.get("sql", "").strip(),
                    parameters=dict(query.get("parameters", {})),
                    post_llm=bool(query.get("post_llm", False)),
                    result_mappings=dict(query.get("result_mappings", {})),
                    required_inputs=tuple(query.get("required_inputs", [])),
                    param_types=dict(query.get("param_types", {})),
                    plans=tuple(self._parse_query_plans(query.get("plans", []))),
                    coalesce=as_bool(query.get("coalesce", True)),
                )
                for query in node.get("db_queries", [])
            )
            parsed[node_id] = Node(
                id=node_id,
                type=node.get("type", "inference"),
                engine=node.get("engine"),
                model=node.get("model"),
                system_prompt=node.get("system_prompt"),
                inputs=tuple(node.get("inputs", [])),
                outputs=tuple(node.get("outputs", [])),
                db_queries=db_queries,
                raw=dict(node),
            )
        return parsed

    def _parse_query_plans(self, plans: Iterable[Dict[str, Any]]) -> List[QueryPlanOption]:
        parsed: List[QueryPlanOption] = []
        for idx, plan in enumerate(plans):
            plan_id = plan.get("id") or f"plan_{idx}"
            description = plan.get("description", plan_id)
            settings = dict(plan.get("settings", {}))
            parsed.append(QueryPlanOption(id=plan_id, description=description, settings=settings))
        return parsed

    def _parse_edges(self, edges: Iterable[Dict[str, Any]]) -> List[Edge]:
        parsed_edges: List[Edge] = []
        for edge in edges:
            source = edge.get("from")
            target = edge.get("to")
            mapping = dict(edge.get("mapping", {}))
            if not source or not target:
                raise GraphValidationError("Edge entries must have 'from' and 'to'.")
            parsed_edges.append(Edge(source=source, target=target, mapping=mapping))
        return parsed_edges

    def _expand_embedded_calls(
        self,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> Tuple[Dict[str, Node], List[Edge]]:
        """Split DB queries and tool calls embedded in a node into standalone CPU nodes.

        A node may declare ``db_queries`` and ``tool_calls`` (entries with a ``name``,
        a ``kind`` of ``http`` or ``processor``, and that tool's own fields, e.g.
        ``sleep_s`` / ``url`` or ``processor`` / ``config``). For each of them:
        - a pre-LLM call (default) becomes a node fed by the original node's parents
          and feeding the node, whose inputs gain the call's outputs;
        - a ``post_llm: true`` call becomes a node that depends on the node and
          consumes its outputs (listed first, so processors that read their first
          input see the LLM output).
        Pre-LLM calls depend only on the node's own parents, never on each other;
        a tool call may set its own ``inputs``.
        """
        new_nodes: Dict[str, Node] = {}
        new_edges: List[Edge] = list(edges)

        incoming: Dict[str, List[Edge]] = {}
        for edge in edges:
            incoming.setdefault(edge.target, []).append(edge)

        def unique_node_id(base: str) -> str:
            candidate = base
            suffix = 1
            while candidate in nodes or candidate in new_nodes:
                candidate = f"{base}_{suffix}"
                suffix += 1
            return candidate

        for node_id, node in nodes.items():
            calls = node.raw.get("tool_calls") or []
            if not node.db_queries and not calls:
                new_nodes[node_id] = node
                continue
            pre_calls: List[Dict[str, Any]] = []
            post_calls: List[Dict[str, Any]] = []
            for call in calls:
                if not isinstance(call, dict) or not call.get("name"):
                    raise GraphValidationError(f"Node '{node_id}': every tool call needs a 'name'.")
                if call.get("kind", "http") not in ("http", "processor"):
                    raise GraphValidationError(
                        f"Node '{node_id}': tool call kind must be 'http' or 'processor' (got {call.get('kind')!r})."
                    )
                (post_calls if as_bool(call.get("post_llm", False)) else pre_calls).append(call)

            pre_queries = [q for q in node.db_queries if not q.post_llm]
            post_queries = [q for q in node.db_queries if q.post_llm]

            # The LLM node's inputs gain the pre-LLM outputs so prompts can consume them.
            added_inputs: List[str] = []
            for q in pre_queries:
                added_inputs.append(q.name)
                added_inputs.extend(q.result_mappings.keys())
            added_inputs.extend(c["name"] for c in pre_calls)
            new_nodes[node_id] = Node(
                id=node_id,
                type=node.type,
                engine=node.engine,
                model=node.model,
                system_prompt=node.system_prompt,
                inputs=tuple(dict.fromkeys(list(node.inputs) + added_inputs)),
                outputs=node.outputs,
                db_queries=tuple(),  # DB work is split out
                raw={k: v for k, v in node.raw.items() if k != "tool_calls"},
            )

            def add_pre(cpu_node: Node) -> None:
                new_nodes[cpu_node.id] = cpu_node
                for edge in incoming.get(node_id, []):
                    new_edges.append(Edge(source=edge.source, target=cpu_node.id, mapping=dict(edge.mapping)))
                new_edges.append(Edge(source=cpu_node.id, target=node_id, mapping={}))

            def add_post(cpu_node: Node) -> None:
                new_nodes[cpu_node.id] = cpu_node
                new_edges.append(Edge(source=node_id, target=cpu_node.id, mapping={}))

            def db_node(query: DBQuery, stage: str, inputs: Tuple[str, ...]) -> Node:
                return Node(
                    id=unique_node_id(f"{node_id}__{stage}__{query.name}"),
                    type="db_query",
                    engine="db",
                    model=None,
                    system_prompt=None,
                    inputs=inputs,
                    outputs=tuple(dict.fromkeys([query.name, *query.result_mappings.keys()])),
                    db_queries=(query,),
                    raw={"parent": node_id, "source": f"{stage}_db", "query": query.name},
                )

            def tool_node(call: Dict[str, Any], stage: str, inputs: Tuple[str, ...]) -> Node:
                kind = call.get("kind", "http")
                fields = {k: v for k, v in call.items() if k not in ("name", "kind", "post_llm", "inputs")}
                return Node(
                    id=unique_node_id(f"{node_id}__{stage}__{call['name']}"),
                    type=kind,
                    engine="http" if kind == "http" else None,
                    model=None,
                    system_prompt=None,
                    inputs=tuple(call["inputs"]) if call.get("inputs") else inputs,
                    outputs=(call["name"],),
                    db_queries=tuple(),
                    raw={**fields, "parent": node_id, "source": f"{stage}_tool"},
                )

            for query in pre_queries:
                add_pre(db_node(query, "pre", node.inputs))
            for call in pre_calls:
                add_pre(tool_node(call, "pre", node.inputs))
            for query in post_queries:
                add_post(db_node(query, "post", node.inputs + node.outputs))
            for call in post_calls:
                add_post(tool_node(call, "post", tuple(dict.fromkeys(node.outputs + node.inputs))))

        return new_nodes, new_edges
