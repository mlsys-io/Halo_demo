"""Database executors and helpers for resolving DBQuery parameters."""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from queue import Empty, Full, LifoQueue
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    TypedDict,
)

try:
    import psycopg
    from psycopg.rows import dict_row
except Exception:  # pragma: no cover - optional dependency
    psycopg = None  # type: ignore[assignment]
    dict_row = None  # type: ignore[assignment]

from .models import DBQuery, PlanMetric
from .utils import MISSING, lookup_path, render_template

LOGGER = logging.getLogger(__name__)

_CONNECT_ENV_MAP = {
    "host": "HALO_PG_HOST",
    "port": "HALO_PG_PORT",
    "user": "HALO_PG_USER",
    "password": "HALO_PG_PASSWORD",
    "dbname": "HALO_PG_DBNAME",
}


class DBResult(TypedDict, total=False):
    query: str
    sql: str
    executed_sql: str
    parameters: Dict[str, Any]
    rows: Sequence[Any]
    rowcount: int
    note: str


class MissingQueryInputs(RuntimeError):
    """Raised when a DBQuery references context inputs that are missing/None."""

    def __init__(self, query_name: str, missing_keys: Sequence[str]):
        self.query_name = query_name
        self.missing_keys = list(missing_keys)
        message = f"Missing required parameters for query {query_name}: {', '.join(self.missing_keys)}"
        super().__init__(message)


class DatabaseExecutor(Protocol):
    def run(
        self,
        query: DBQuery,
        context: Mapping[str, Any],
        *,
        node_id: str | None = None,
    ) -> Any: ...


class _BaseDatabaseExecutor:
    def _format_sql(self, query: DBQuery) -> str:
        return query.sql

    def _resolve_parameters(
        self,
        query: DBQuery,
        context: Mapping[str, Any],
        *,
        node_id: str | None = None,
    ) -> Tuple[Dict[str, Any], Sequence[str]]:
        return resolve_query_parameters(query, context, node_id=node_id)

    def _log_result(
        self,
        node_id: str | None,
        query_name: str,
        result: Mapping[str, Any],
    ) -> None:
        LOGGER.debug(
            "Executed query %s (node=%s) rows=%s",
            query_name,
            node_id or "unknown",
            result.get("rowcount", "?"),
        )


class DefaultDatabaseExecutor(_BaseDatabaseExecutor):
    """Fallback executor that simply records the intended SQL."""

    def run(
        self,
        query: DBQuery,
        context: Mapping[str, Any],
        *,
        node_id: str | None = None,
    ) -> Dict[str, Any]:
        params, missing = self._resolve_parameters(query, context, node_id=node_id)
        if missing:
            raise MissingQueryInputs(query.name, missing)
        sql_text = self._format_sql(query)
        return _build_result_dict(
            query,
            sql_text=sql_text,
            executed_sql=sql_text,
            parameters=params,
            note="Database executor not configured. This is a dry-run result.",
        )


class PostgresDatabaseExecutor(_BaseDatabaseExecutor):
    """Executes DBQuery objects against PostgreSQL using psycopg3."""

    _PARAM_PATTERN = re.compile(r"(?<!:):([a-zA-Z_][\w]*)")
    _PYFORMAT_PATTERN = re.compile(r"%\([^)]+\)[sbt]")

    def __init__(
        self,
        dsn: str | None = None,
        *,
        autocommit: bool = True,
        connect_kwargs: Dict[str, Any] | None = None,
        pool_size: int = 5,
        prepare_threshold: int | None = 1,
    ) -> None:
        if psycopg is None:  # pragma: no cover - import guard
            raise RuntimeError(
                "psycopg is required for PostgresDatabaseExecutor. "
                "Install halo-dev with the 'psycopg' extra or add psycopg[binary]."
            )
        self.dsn = dsn or os.getenv("HALO_PG_DSN")
        self.autocommit = autocommit
        self.connect_kwargs = connect_kwargs.copy() if connect_kwargs else {}
        env_threshold = os.getenv("HALO_PG_PREPARE_THRESHOLD")
        if env_threshold is not None:
            try:
                prepare_threshold = int(env_threshold)
            except ValueError:
                pass
        self.prepare_threshold = prepare_threshold
        self._pool = (
            _ConnectionPool(self._create_connection, max(0, pool_size))
            if pool_size > 0
            else None
        )

    def run(
        self,
        query: DBQuery,
        context: Mapping[str, Any],
        *,
        node_id: str | None = None,
    ) -> Dict[str, Any]:
        sql = self._format_sql(query)
        params, missing = self._resolve_parameters(query, context, node_id=node_id)
        if missing:
            raise MissingQueryInputs(query.name, missing)

        with self._connection() as conn:
            if self.autocommit:
                conn.autocommit = True
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
                result = _build_result_dict(
                    query,
                    sql_text=query.sql,
                    executed_sql=sql,
                    parameters=params,
                    rows=rows,
                    rowcount=cur.rowcount,
                )
                self._log_result(node_id, query.name, result)
                return result

    def _format_sql(self, query: DBQuery) -> str:
        """Convert :named parameters to psycopg's %(name)s style."""
        sql = self._PARAM_PATTERN.sub(lambda match: f"%({match.group(1)})s", query.sql)
        return self._escape_percent_literals(sql)

    def _escape_percent_literals(self, sql: str) -> str:
        """Escape literal % so psycopg doesn't treat them as placeholders."""
        if "%" not in sql:
            return sql
        parts: List[str] = []
        idx = 0
        length = len(sql)
        while idx < length:
            ch = sql[idx]
            if ch != "%":
                parts.append(ch)
                idx += 1
                continue
            if idx + 1 < length and sql[idx + 1] == "%":
                parts.append("%%")
                idx += 2
                continue
            if idx + 1 < length and sql[idx + 1] == "(":
                match = self._PYFORMAT_PATTERN.match(sql, idx)
                if match:
                    parts.append(match.group(0))
                    idx = match.end()
                    continue
            if idx + 1 < length and sql[idx + 1] in ("s", "b", "t"):
                parts.append(sql[idx : idx + 2])
                idx += 2
                continue
            parts.append("%%")
            idx += 1
        return "".join(parts)

    @contextmanager
    def _connection(self) -> Iterator["psycopg.Connection[Any]"]:
        if self._pool is None:
            conn = self._create_connection()
            try:
                yield conn
            finally:
                conn.close()
            return
        with self._pool.connection() as pooled:
            yield pooled

    def _create_connection(self) -> "psycopg.Connection[Any]":
        kwargs = self._effective_connect_kwargs()
        if self.prepare_threshold is not None:
            kwargs.setdefault("prepare_threshold", self.prepare_threshold)
        if self.dsn:
            return psycopg.connect(self.dsn, row_factory=dict_row, **kwargs)
        if not kwargs:
            raise RuntimeError(
                "PostgresDatabaseExecutor requires either a DSN or explicit connection kwargs."
            )
        return psycopg.connect(row_factory=dict_row, **kwargs)

    def _effective_connect_kwargs(self) -> Dict[str, Any]:
        kwargs = dict(self.connect_kwargs)
        for key, env in _CONNECT_ENV_MAP.items():
            if key in kwargs:
                continue
            value = os.getenv(env)
            if value is None:
                continue
            if key == "port":
                try:
                    kwargs[key] = int(value)
                except ValueError:
                    continue
            else:
                kwargs[key] = value
        return kwargs


class PostgresPlanExplainer(PostgresDatabaseExecutor):
    """Runs EXPLAIN with optional planner settings."""

    def __init__(
        self,
        dsn: str | None = None,
        *,
        autocommit: bool = True,
        connect_kwargs: Dict[str, Any] | None = None,
        pool_size: int = 5,
        analyze: bool = False,
    ) -> None:
        super().__init__(
            dsn,
            autocommit=autocommit,
            connect_kwargs=connect_kwargs,
            pool_size=pool_size,
        )
        self.analyze = analyze

    def explain(
        self,
        query: DBQuery,
        params: Mapping[str, Any],
        *,
        settings: Mapping[str, Any] | None = None,
    ) -> tuple[Any, PlanMetric]:
        sql = self._format_sql(query)
        options = ["FORMAT JSON", "COSTS TRUE"]
        if self.analyze:
            options.insert(0, "BUFFERS")
            options.insert(0, "ANALYZE")
        explain_sql = f"EXPLAIN ({', '.join(options)}) {sql}"
        with self._connection() as conn:
            with conn.cursor() as cur:
                reset_keys: List[str] = []
                if settings:
                    for key, value in settings.items():
                        self._apply_setting(cur, key, value)
                        reset_keys.append(key)
                cur.execute(explain_sql, params)
                rows = cur.fetchall()
                plan_json = None
                if rows:
                    first_row = rows[0]
                    if isinstance(first_row, Mapping):
                        plan_json = first_row.get("QUERY PLAN")
                        if plan_json is None and first_row:
                            plan_json = next(iter(first_row.values()))
                    elif isinstance(first_row, Sequence):
                        plan_json = first_row[0] if first_row else None
                    else:
                        plan_json = first_row
                for key in reset_keys:
                    cur.execute(f"RESET {key}")
                plan_json, metrics = self._extract_plan_metrics(plan_json)
                return plan_json, metrics

    _SETTING_KEY_RE = re.compile(r"^[a-zA-Z_][\w.]*$")

    def _apply_setting(self, cursor: "psycopg.Cursor[Any]", key: str, value: Any) -> None:
        if not self._SETTING_KEY_RE.match(key):
            raise ValueError(f"Unsafe setting key: {key}")
        literal = self._format_setting_value(value)
        cursor.execute(f"SET {key} = {literal}")

    def _format_setting_value(self, value: Any) -> str:
        if isinstance(value, bool):
            return "on" if value else "off"
        if value is None:
            return "DEFAULT"
        if isinstance(value, (int, float)):
            return str(value)
        escaped = str(value).replace("'", "''")
        return f"'{escaped}'"

    def _extract_plan_metrics(self, plan_json: Any) -> tuple[Any, PlanMetric]:
        runtime: float | None = None
        shared_hit = shared_read = local_hit = local_read = None
        relations: Dict[str, int] = {}
        indexes: Dict[str, int] = {}
        try:
            if isinstance(plan_json, list) and plan_json:
                top = plan_json[0]
                runtime_val = top.get("Execution Time")
                if runtime_val is not None:
                    runtime = float(runtime_val)
                plan = top.get("Plan")
                if isinstance(plan, Mapping):
                    agg = self._aggregate_buffer_metrics(plan)
                    shared_hit, shared_read, local_hit, local_read, rels, idxs = agg
                    relations.update(rels)
                    indexes.update(idxs)
        except Exception:
            pass
        metric = PlanMetric(
            runtime_ms=runtime,
            shared_hit_blocks=shared_hit,
            shared_read_blocks=shared_read,
            local_hit_blocks=local_hit,
            local_read_blocks=local_read,
            relations=relations,
            indexes=indexes,
        )
        return plan_json, metric

    def _aggregate_buffer_metrics(
        self,
        node: Mapping[str, Any],
    ) -> tuple[int | None, int | None, int | None, int | None, Dict[str, int], Dict[str, int]]:
        def safe_int(value: Any) -> int:
            return int(value) if value is not None else 0

        shared_hit = safe_int(node.get("Shared Hit Blocks"))
        shared_read = safe_int(node.get("Shared Read Blocks"))
        local_hit = safe_int(node.get("Local Hit Blocks"))
        local_read = safe_int(node.get("Local Read Blocks"))
        relations: Dict[str, int] = {}
        indexes: Dict[str, int] = {}
        relation = node.get("Relation Name")
        index_name = node.get("Index Name")
        footprint = safe_int(node.get("Shared Hit Blocks")) + safe_int(node.get("Shared Read Blocks"))
        if relation:
            relations[relation] = relations.get(relation, 0) + footprint
        if index_name:
            indexes[index_name] = indexes.get(index_name, 0) + footprint
        for child in node.get("Plans", []) or []:
            if isinstance(child, Mapping):
                c_hit, c_read, c_lhit, c_lread, c_rel, c_idx = self._aggregate_buffer_metrics(child)
                shared_hit += c_hit or 0
                shared_read += c_read or 0
                local_hit += c_lhit or 0
                local_read += c_lread or 0
                for name, value in c_rel.items():
                    relations[name] = relations.get(name, 0) + value
                for name, value in c_idx.items():
                    indexes[name] = indexes.get(name, 0) + value
        return shared_hit, shared_read, local_hit, local_read, relations, indexes


class _ConnectionPool:
    """Minimal LIFO connection pool shared per worker process."""

    def __init__(self, factory: Callable[[], "psycopg.Connection[Any]"], max_size: int):
        self._factory = factory
        self._pool: LifoQueue["psycopg.Connection[Any]"] = LifoQueue(maxsize=max_size or 0)

    @contextmanager
    def connection(self) -> Iterator["psycopg.Connection[Any]"]:
        conn = self.acquire()
        try:
            yield conn
        finally:
            self.release(conn)

    def acquire(self) -> "psycopg.Connection[Any]":
        try:
            conn = self._pool.get_nowait()
            if conn.closed:
                return self._factory()
            return conn
        except Empty:
            return self._factory()

    def release(self, conn: "psycopg.Connection[Any]") -> None:
        if conn.closed:
            return
        try:
            self._pool.put_nowait(conn)
        except Full:
            conn.close()


def resolve_query_parameters(
    query: DBQuery,
    context: Mapping[str, Any],
    *,
    node_id: str | None = None,
) -> Tuple[Dict[str, Any], List[str]]:
    """Render query parameters from context and track missing inputs."""
    missing: List[str] = []
    seen: set[str] = set()

    def register_missing(key: str, template: str | None = None) -> None:
        if key in seen:
            return
        seen.add(key)
        missing.append(key)
        detail = f" in template '{template}'" if template else ""
        LOGGER.warning(
            "Missing value for '%s' in query %s%s%s",
            key,
            query.name,
            f" (node={node_id})" if node_id else "",
            detail,
        )

    for required in query.required_inputs:
        value = lookup_path(context, required, default=MISSING)
        if value is MISSING or value is None:
            register_missing(required)

    resolved: Dict[str, Any] = {}
    for key, template in query.parameters.items():
        raw_value = _resolve_value(template, context, register_missing)
        typed_value = _apply_param_type(raw_value, query.param_types.get(key))
        resolved[key] = typed_value

    return resolved, missing


def _build_result_dict(
    query: DBQuery,
    *,
    sql_text: str | None,
    executed_sql: str | None,
    parameters: Mapping[str, Any],
    rows: Sequence[Any] | None = None,
    rowcount: int | None = None,
    note: str | None = None,
) -> DBResult:
    """Create a consistent result dictionary for DB executor outputs."""
    result: DBResult = {
        "query": query.name,
        "sql": sql_text or query.sql,
        "parameters": dict(parameters),
    }
    if executed_sql is not None:
        result["executed_sql"] = executed_sql
    if rows is not None:
        result["rows"] = list(rows)
    if rowcount is not None:
        result["rowcount"] = rowcount
    if note is not None:
        result["note"] = note
    return result


def _resolve_value(
    value: Any,
    context: Mapping[str, Any],
    register_missing: Callable[[str, str | None], None],
) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if _is_pure_placeholder(stripped):
            ctx_key = stripped[2:-2].strip()
            resolved = lookup_path(context, ctx_key, default=MISSING)
            if resolved is MISSING or resolved is None:
                register_missing(ctx_key, None)
                return None
            return resolved
        rendered = render_template(value, context, on_missing=register_missing)
        return _maybe_coerce(rendered)
    if isinstance(value, Mapping):
        return {k: _resolve_value(v, context, register_missing) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_resolve_value(item, context, register_missing) for item in value]
    return value


def _apply_param_type(value: Any, type_hint: str | None) -> Any:
    if value is None or not type_hint:
        return value
    hint = type_hint.lower()
    if hint == "json":
        if isinstance(value, (dict, list)):
            return value
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Failed to parse JSON parameter: {value}") from exc
        return value
    if hint.endswith("[]"):
        base_hint = hint[:-2]
        sequence = _ensure_sequence(value)
        return [_coerce_scalar(item, base_hint) for item in sequence]
    return _coerce_scalar(value, hint)


def _ensure_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, (list, tuple)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


_SCALAR_CASTERS: Dict[str, Callable[[Any], Any]] = {
    "int": int,
    "integer": int,
    "float": float,
    "double": float,
    "text": str,
    "string": str,
    "str": str,
}


def _coerce_scalar(value: Any, hint: str) -> Any:
    if value is None:
        return None
    normalized = hint.lower()
    if normalized in {"bool", "boolean"}:
        return _coerce_bool(value)
    caster = _SCALAR_CASTERS.get(normalized)
    if caster is not None:
        return caster(value)
    return value


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1"}:
            return True
        if lowered in {"false", "0"}:
            return False
    return bool(value)


def _is_pure_placeholder(value: str) -> bool:
    return value.startswith("{{") and value.endswith("}}")


def _maybe_coerce(value: str) -> Any:
    trimmed = value.strip()
    if not trimmed:
        return value
    if trimmed.lower() in {"true", "false"}:
        return trimmed.lower() == "true"
    try:
        if "." in trimmed:
            return float(trimmed)
        return int(trimmed)
    except ValueError:
        pass
    if (trimmed.startswith("[") and trimmed.endswith("]")) or (
        trimmed.startswith("{") and trimmed.endswith("}")
    ):
        try:
            return json.loads(trimmed)
        except json.JSONDecodeError:
            return value
    return value


def make_peer_postgres_executor(
    *,
    dbname: str,
    user: str,
    host: str,
port: int,
    pool_size: int | None = None,
) -> PostgresDatabaseExecutor:
    connect_kwargs = {
        "dbname": dbname,
        "user": user,
        "host": host,
        "port": port,
    }
    effective_pool = max(1, pool_size) if pool_size is not None else None
    kwargs: Dict[str, Any] = {"connect_kwargs": connect_kwargs}
    if effective_pool is not None:
        kwargs["pool_size"] = effective_pool
    return PostgresDatabaseExecutor(**kwargs)
