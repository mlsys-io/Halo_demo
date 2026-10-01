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
        nodes, edges = self._expand_db_queries(nodes, edges)
        nodes, edges = self._expand_tool_calls(nodes, edges)
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

    def _expand_db_queries(
        self,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> Tuple[Dict[str, Node], List[Edge]]:
        """Split each node's DB queries into standalone CPU nodes.

        For a node with DB queries:
        - Create one DB node per pre-LLM query (post_llm=False), and add dependency edges
          from the original parents to the DB node, then DB node -> LLM node.
        - Create one DB node per post-LLM query (post_llm=True), dependent on the LLM node.
        - The original node becomes a pure LLM node (db_queries cleared) and its inputs
          are augmented with DB output names so prompts can consume them.
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
            if not node.db_queries:
                new_nodes[node_id] = node
                continue

            pre_queries = [q for q in node.db_queries if not q.post_llm]
            post_queries = [q for q in node.db_queries if q.post_llm]

            # Augment LLM inputs to include DB outputs for prompt-building.
            db_output_names: List[str] = []
            for q in pre_queries:
                db_output_names.append(q.name)
                db_output_names.extend(q.result_mappings.keys())
            llm_inputs = tuple(dict.fromkeys(list(node.inputs) + db_output_names))
            llm_node = Node(
                id=node_id,
                type=node.type,
                engine=node.engine,
                model=node.model,
                system_prompt=node.system_prompt,
                inputs=llm_inputs,
                outputs=node.outputs,
                db_queries=tuple(),  # DB work is split out
                raw=dict(node.raw),
            )
            new_nodes[node_id] = llm_node

            # Pre-LLM DB queries become standalone CPU nodes feeding the LLM.
            for query in pre_queries:
                db_node_id = unique_node_id(f"{node_id}__pre__{query.name}")
                outputs = tuple(dict.fromkeys([query.name, *query.result_mappings.keys()]))
                db_node = Node(
                    id=db_node_id,
                    type="db_query",
                    engine="db",
                    model=None,
                    system_prompt=None,
                    inputs=node.inputs,
                    outputs=outputs,
                    db_queries=(query,),
                    raw={"parent": node_id, "source": "pre_db", "query": query.name},
                )
                new_nodes[db_node_id] = db_node

                for edge in incoming.get(node_id, []):
                    new_edges.append(Edge(source=edge.source, target=db_node_id, mapping=dict(edge.mapping)))
                new_edges.append(Edge(source=db_node_id, target=node_id, mapping={}))

            # Post-LLM DB queries depend on the LLM node.
            for query in post_queries:
                db_node_id = unique_node_id(f"{node_id}__post__{query.name}")
                outputs = tuple(dict.fromkeys([query.name, *query.result_mappings.keys()]))
                db_node = Node(
                    id=db_node_id,
                    type="db_query",
                    engine="db",
                    model=None,
                    system_prompt=None,
                    inputs=node.inputs + node.outputs,
                    outputs=outputs,
                    db_queries=(query,),
                    raw={"parent": node_id, "source": "post_db", "query": query.name},
                )
                new_nodes[db_node_id] = db_node
                new_edges.append(Edge(source=node_id, target=db_node_id, mapping={}))

        return new_nodes, new_edges

    def _expand_tool_calls(
        self,
        nodes: Dict[str, Node],
        edges: List[Edge],
    ) -> Tuple[Dict[str, Node], List[Edge]]:
        """Split API / local-function calls embedded in a node into standalone tool nodes.

        A node may declare ``tool_calls``: entries with a ``name`` and ``kind``
        (``http`` or ``processor``) plus that tool's own fields (e.g. ``sleep_s`` /
        ``url`` for HTTP, ``processor`` / ``config`` for local functions). Like
        ``db_queries``, pre-LLM calls (default) feed the node, ``post_llm: true``
        calls consume its outputs, and the node's inputs gain the call outputs.
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
            if not calls:
                new_nodes[node_id] = node
                continue
            for call in calls:
                if not isinstance(call, dict) or not call.get("name"):
                    raise GraphValidationError(f"Node '{node_id}': every tool call needs a 'name'.")
                if call.get("kind", "http") not in ("http", "processor"):
                    raise GraphValidationError(
                        f"Node '{node_id}': tool call kind must be 'http' or 'processor' (got {call.get('kind')!r})."
                    )

            pre_calls = [c for c in calls if not c.get("post_llm", False)]
            post_calls = [c for c in calls if c.get("post_llm", False)]
            raw = {k: v for k, v in node.raw.items() if k != "tool_calls"}
            new_nodes[node_id] = Node(
                id=node_id,
                type=node.type,
                engine=node.engine,
                model=node.model,
                system_prompt=node.system_prompt,
                inputs=tuple(dict.fromkeys(list(node.inputs) + [c["name"] for c in pre_calls])),
                outputs=node.outputs,
                db_queries=node.db_queries,
                raw=raw,
            )

            def tool_node(call: Dict[str, Any], stage: str, inputs: Tuple[str, ...]) -> Node:
                kind = call.get("kind", "http")
                fields = {k: v for k, v in call.items() if k not in ("name", "kind", "post_llm")}
                return Node(
                    id=unique_node_id(f"{node_id}__{stage}__{call['name']}"),
                    type=kind,
                    engine="http" if kind == "http" else None,
                    model=None,
                    system_prompt=None,
                    inputs=inputs,
                    outputs=(call["name"],),
                    db_queries=tuple(),
                    raw={**fields, "parent": node_id, "source": f"{stage}_tool"},
                )

            # Pre-LLM calls become standalone tool nodes feeding the LLM node.
            for call in pre_calls:
                tnode = tool_node(call, "pre", tuple(node.inputs))
                new_nodes[tnode.id] = tnode
                for edge in incoming.get(node_id, []):
                    new_edges.append(Edge(source=edge.source, target=tnode.id, mapping=dict(edge.mapping)))
                new_edges.append(Edge(source=tnode.id, target=node_id, mapping={}))

            # Post-LLM calls depend on the LLM node.
            for call in post_calls:
                tnode = tool_node(call, "post", tuple(node.inputs) + tuple(node.outputs))
                new_nodes[tnode.id] = tnode
                new_edges.append(Edge(source=node_id, target=tnode.id, mapping={}))

        return new_nodes, new_edges
