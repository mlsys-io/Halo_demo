#!/usr/bin/env python3
"""Local mock API for the HTTP operators of the evaluation templates (templates/exps).

The paper's HTTP tool calls stand for remote lookups. This service answers them
from the same Postgres databases the templates' SQL operators use, so every HTTP
node returns real data that its downstream prompt consumes:

  IMDb movie-metadata API (W1, W2)
    GET /imdb/titles/search?keyword=K&limit=25           titles matching K, most-voted first
    GET /imdb/titles/recent?keyword=K&since=2000&limit=15 movies since 2000 matching K
    GET /imdb/people/search?keyword=K&limit=40            people whose name contains K
  Wikipedia (FineWiki) category and link API (W3, W4)
    GET /finewiki/categories?topic=T&limit=30            categories of the articles matching T
    GET /finewiki/outbound-links?topic=T&limit=40        outbound links of the articles matching T
  TPC-H report API (W5, W6)
    GET /tpch/reports/shipmode-priority?shipmode=MAIL&shipmode2=SHIP&year=1994   (Q12)
    GET /tpch/reports/promo-revenue?year=1994                                     (Q14)
    GET /tpch/reports/segment-revenue?segment=BUILDING&year=1994
  GET /health

Every response is a JSON object whose "rows" hold the result, so Halo renders it in
a prompt like a SQL result. Responses are deterministic: the IMDb and FineWiki
endpoints run the SQL their template node replaced (with a unique tiebreaker added
to ORDER BY) and cache each answer; the TPC-H reports are aggregated once at
startup over every parameter value, giving the same numbers as running the
replaced query per request without its full lineitem scan on every call.

The service adds no latency of its own: an HTTP node declaring ``url`` and
``latency_s`` issues the request and Halo pads the call to a Gamma-distributed
latency with that mean (see HTTPNodeExecutor). A call whose answer takes longer
than its sampled latency is not cut short.

Usage (from the repository root; needs ``uv sync --extra postgres``):
    python scripts/mock_api.py --port 8765 --pg-host localhost --pg-user postgres
Postgres settings default to HALO_PG_HOST / HALO_PG_PORT / HALO_PG_USER /
HALO_PG_PASSWORD (or libpq's defaults); --datasets serves a subset.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
import time
import urllib.parse
from datetime import date, datetime
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a checkout

from halo import db as halo_db  # noqa: E402
from halo.models import DBQuery  # noqa: E402

LOGGER = logging.getLogger("mock_api")
DATASETS = ("imdb", "finewiki", "tpch")
# Tables each dataset's endpoints read; probed at startup.
REQUIRED_TABLES = {
    "imdb": ("title_basics", "title_ratings", "name_basics"),
    "finewiki": ("pages",),
    "tpch": ("lineitem", "orders", "customer", "part"),
}

# ---------------------------------------------------------------------------
# SQL. The IMDb and FineWiki statements are the ones the HTTP nodes replaced in
# the research templates; LIMIT is a parameter and ORDER BY gains a unique key.
# ---------------------------------------------------------------------------

IMDB_TITLE_SEARCH = """
SELECT b.tconst, b.primary_title, b.original_title, b.start_year, b.title_type,
       r.average_rating, r.num_votes
FROM title_basics AS b
LEFT JOIN title_ratings AS r ON r.tconst = b.tconst
WHERE b.title_type IN ('movie','tvMovie','tvSeries')
  AND (b.primary_title ILIKE '%%' || :keyword || '%%'
       OR b.original_title ILIKE '%%' || :keyword || '%%')
ORDER BY COALESCE(r.num_votes, 0) DESC, b.tconst
LIMIT :limit
"""

IMDB_TITLE_RECENT = """
SELECT b.tconst, b.primary_title, b.start_year, b.genres, r.average_rating, r.num_votes
FROM title_basics AS b
LEFT JOIN title_ratings AS r ON r.tconst = b.tconst
WHERE b.title_type IN ('movie','tvMovie')
  AND b.start_year >= :since
  AND (b.primary_title ILIKE '%%' || :keyword || '%%'
       OR b.original_title ILIKE '%%' || :keyword || '%%')
ORDER BY COALESCE(r.num_votes, 0) DESC, b.tconst
LIMIT :limit
"""

IMDB_PEOPLE_SEARCH = """
SELECT n.nconst, n.primary_name, n.primary_profession, n.known_for_titles
FROM name_basics AS n
WHERE n.primary_name ILIKE '%%' || :keyword || '%%'
ORDER BY n.primary_name, n.nconst
LIMIT :limit
"""

# English articles whose title contains the topic (W3's replaced outbound-link query).
FINEWIKI_TOPIC_PAGES = """
SELECT page_id, title, url, wikitext
FROM pages
WHERE in_language = 'en'
  AND title ILIKE '%%' || :topic || '%%'
ORDER BY page_id ASC
LIMIT :limit
"""
# The category API scans the same page pool. (The replaced category query matched
# titles starting with "Category:", which FineWiki, articles only, never contains.)
CATEGORY_PAGE_POOL = 40

# TPC-H report aggregates, one row per parameter combination.
TPCH_Q12_BY_YEAR_MODE = """
SELECT EXTRACT(YEAR FROM l.l_receiptdate)::int AS year,
       l.l_shipmode,
       COUNT(*) AS line_count,
       SUM(CASE WHEN o.o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_line_count,
       SUM(CASE WHEN o.o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_line_count
FROM orders o
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE l.l_commitdate < l.l_receiptdate
  AND l.l_shipdate < l.l_commitdate
GROUP BY 1, 2
"""

TPCH_Q14_BY_YEAR = """
SELECT EXTRACT(YEAR FROM l.l_shipdate)::int AS year,
       100.00 * SUM(CASE WHEN p.p_type LIKE 'PROMO%' THEN l.l_extendedprice * (1 - l.l_discount) ELSE 0 END)
         / SUM(l.l_extendedprice * (1 - l.l_discount)) AS promo_revenue
FROM lineitem l
JOIN part p ON p.p_partkey = l.l_partkey
WHERE EXTRACT(MONTH FROM l.l_shipdate) = 9
GROUP BY 1
"""

TPCH_SEGMENT_BY_YEAR = """
SELECT c.c_mktsegment AS segment,
       EXTRACT(YEAR FROM o.o_orderdate)::int AS year,
       SUM(l.l_extendedprice * (1 - l.l_discount)) AS revenue,
       COUNT(*) AS line_cnt,
       SUM(CASE WHEN o.o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS high_pri_lines,
       SUM(CASE WHEN o.o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) AS low_pri_lines
FROM orders o
JOIN customer c ON c.c_custkey = o.o_custkey
JOIN lineitem l ON l.l_orderkey = o.o_orderkey
WHERE l.l_shipdate > o.o_orderdate
GROUP BY 1, 2
"""


class BadRequest(ValueError):
    """Invalid request parameters (HTTP 400)."""


class Unavailable(RuntimeError):
    """The dataset is not served or its database cannot be reached (HTTP 503)."""


class _SingleFlightCache:
    """Answers by key; concurrent requests for a missing key wait for one computation."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: Dict[Any, Any] = {}
        self._pending: Dict[Any, threading.Event] = {}

    def get(self, key: Any, compute: Callable[[], Any]) -> Any:
        while True:
            with self._lock:
                if key in self._values:
                    return self._values[key]
                event = self._pending.get(key)
                if event is None:
                    event = self._pending[key] = threading.Event()
                    break
            event.wait()  # another thread computes it; on failure, retry
        try:
            value = compute()
            with self._lock:
                self._values[key] = value
            return value
        finally:
            with self._lock:
                self._pending.pop(key, None)
            event.set()


# ---------------------------------------------------------------------------
# Wikitext helpers
# ---------------------------------------------------------------------------

_WIKILINK = re.compile(r"\[\[([^\[\]|#]+)(?:#[^\[\]|]*)?(?:\|[^\[\]]*)?\]\]")
_CATEGORY = re.compile(r"\[\[\s*category\s*:\s*([^\[\]|]+?)\s*(?:\|[^\[\]]*)?\]\]", re.IGNORECASE)
_NON_ARTICLE_PREFIXES = {
    "file", "image", "media", "category", "template", "wikipedia", "wp", "help", "portal", "special",
    "draft", "module", "user", "talk", "wikt", "wiktionary", "commons", "meta", "species", "s", "q", "n",
}
_LANGUAGE_PREFIX = re.compile(r"^[a-z]{2,3}(-[a-z]+)*$")


def _article_title(target: str) -> str:
    title = " ".join(target.replace("_", " ").split())
    return title[:1].upper() + title[1:]


def outbound_links(wikitext: str | None) -> List[str]:
    """Distinct article titles a page links to, in page order (no file, category,
    interwiki, or other namespace links)."""
    links: Dict[str, None] = {}
    for match in _WIKILINK.finditer(wikitext or ""):
        target = match.group(1).strip()
        if ":" in target:
            prefix = target.lstrip(":").split(":", 1)[0].strip().lower()
            if target.startswith(":") or prefix in _NON_ARTICLE_PREFIXES or _LANGUAGE_PREFIX.match(prefix):
                continue
        title = _article_title(target)
        if title:
            links.setdefault(title, None)
    return list(links)


def categories(wikitext: str | None) -> List[str]:
    """Distinct categories a page declares with [[Category:...]]."""
    return list(dict.fromkeys(_article_title(name) for name in _CATEGORY.findall(wikitext or "") if name.strip()))


def _category_url(name: str) -> str:
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote("Category:" + name.replace(" ", "_"), safe=":/(),'")


# ---------------------------------------------------------------------------
# Request parameters
# ---------------------------------------------------------------------------

def _text(query: Mapping[str, str], name: str) -> str:
    # An empty value is passed through, as the replaced SQL would receive it.
    if name not in query:
        raise BadRequest(f"missing required parameter '{name}'")
    return query[name]


def _int(query: Mapping[str, str], name: str, default: int | None = None, lo: int | None = None,
         hi: int | None = None) -> int:
    raw = query.get(name)
    if raw is None or not raw.strip():
        if default is None:
            raise BadRequest(f"missing required parameter '{name}'")
        return default
    try:
        value = int(raw)
    except ValueError:
        raise BadRequest(f"parameter '{name}' must be an integer (got {raw!r})") from None
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise BadRequest(f"parameter '{name}' must be within [{lo}, {hi}] (got {value})")
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, str):
        return value.rstrip()  # CHAR(n) padding
    return value


def _clean(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: _jsonable(value) for key, value in row.items()}


def _json_default(value: Any) -> Any:
    converted = _jsonable(value)
    return str(value) if converted is value else converted


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class MockAPI:
    """Routes requests to dataset handlers backed by halo's Postgres executor."""

    def __init__(self, executors: Mapping[str, halo_db.PostgresDatabaseExecutor]) -> None:
        self.executors = dict(executors)
        self._cache = _SingleFlightCache()
        self._tpch: Dict[str, List[Dict[str, Any]]] = {}
        self.routes: Dict[str, tuple[str, Callable[[Mapping[str, str]], Dict[str, Any]]]] = {
            "/imdb/titles/search": ("imdb", self.imdb_title_search),
            "/imdb/titles/recent": ("imdb", self.imdb_title_recent),
            "/imdb/people/search": ("imdb", self.imdb_people_search),
            "/finewiki/categories": ("finewiki", self.finewiki_categories),
            "/finewiki/outbound-links": ("finewiki", self.finewiki_outbound_links),
            "/tpch/reports/shipmode-priority": ("tpch", self.tpch_shipmode_priority),
            "/tpch/reports/promo-revenue": ("tpch", self.tpch_promo_revenue),
            "/tpch/reports/segment-revenue": ("tpch", self.tpch_segment_revenue),
        }

    # -- database access ----------------------------------------------------

    def _rows(self, dataset: str, name: str, sql: str, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
        query = DBQuery(name=name, sql=sql.strip(), parameters={key: f"{{{{ {key} }}}}" for key in params})
        try:
            result = self.executors[dataset].run(query, dict(params), node_id="mock_api")
        except Exception as exc:  # psycopg.OperationalError and friends
            if type(exc).__name__ in ("OperationalError", "InterfaceError"):
                raise Unavailable(f"Postgres database for '{dataset}' is unreachable: {exc}") from exc
            raise
        return [dict(row) for row in result.get("rows", [])]

    def _cached_rows(self, dataset: str, name: str, sql: str, params: Mapping[str, Any]) -> List[Dict[str, Any]]:
        key = (dataset, name, tuple(sorted(params.items())))
        return self._cache.get(key, lambda: self._rows(dataset, name, sql, params))

    def check(self, dataset: str) -> None:
        """Probe the tables an endpoint group reads; raises with a clear message."""
        for table in REQUIRED_TABLES[dataset]:
            self._rows(dataset, f"probe_{table}", f"SELECT 1 FROM {table} LIMIT 1", {})

    def prepare_tpch(self) -> None:
        """Aggregate the TPC-H reports once (one pass per report over lineitem)."""
        for name, sql in (("q12", TPCH_Q12_BY_YEAR_MODE), ("q14", TPCH_Q14_BY_YEAR),
                          ("segment", TPCH_SEGMENT_BY_YEAR)):
            start = time.perf_counter()
            self._tpch[name] = [_clean(row) for row in self._rows("tpch", f"tpch_{name}_report", sql, {})]
            LOGGER.info("tpch: aggregated report %s (%d rows) in %.1fs", name, len(self._tpch[name]),
                        time.perf_counter() - start)

    def handle(self, path: str, query: Mapping[str, str]) -> Dict[str, Any]:
        if path in ("/", "/health"):
            return {"status": "ok", "datasets": sorted(self.executors), "endpoints": sorted(self.routes)}
        route = self.routes.get(path)
        if route is None:
            raise LookupError(path)
        dataset, handler = route
        if dataset not in self.executors:
            raise Unavailable(f"dataset '{dataset}' is not served (start with --datasets including {dataset})")
        return {"endpoint": path, **handler(query)}

    # -- IMDb movie-metadata API -------------------------------------------

    def imdb_title_search(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"keyword": _text(query, "keyword"), "limit": _int(query, "limit", 25, 1, 100)}
        rows = self._cached_rows("imdb", "title_search", IMDB_TITLE_SEARCH, params)
        return {"params": params, "rowcount": len(rows), "rows": [_clean(r) for r in rows]}

    def imdb_title_recent(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"keyword": _text(query, "keyword"), "since": _int(query, "since", 2000),
                  "limit": _int(query, "limit", 15, 1, 100)}
        rows = self._cached_rows("imdb", "title_recent", IMDB_TITLE_RECENT, params)
        return {"params": params, "rowcount": len(rows), "rows": [_clean(r) for r in rows]}

    def imdb_people_search(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"keyword": _text(query, "keyword"), "limit": _int(query, "limit", 40, 1, 100)}
        rows = self._cached_rows("imdb", "people_search", IMDB_PEOPLE_SEARCH, params)
        return {"params": params, "rowcount": len(rows), "rows": [_clean(r) for r in rows]}

    # -- Wikipedia category and link API -----------------------------------

    def _topic_pages(self, topic: str, limit: int) -> List[Dict[str, Any]]:
        return self._cached_rows("finewiki", "topic_pages", FINEWIKI_TOPIC_PAGES, {"topic": topic, "limit": limit})

    def finewiki_categories(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"topic": _text(query, "topic"), "limit": _int(query, "limit", 30, 1, 100)}
        pages = self._topic_pages(params["topic"], CATEGORY_PAGE_POOL)
        members: Dict[str, List[str]] = {}
        for page in pages:
            for name in categories(page.get("wikitext")):
                members.setdefault(name, []).append(page["title"])
        ranked = sorted(members.items(), key=lambda item: (-len(item[1]), item[0]))[: params["limit"]]
        rows = [
            {"category": name, "title": f"Category:{name}", "url": _category_url(name),
             "member_count": len(titles), "member_pages": titles[:10]}
            for name, titles in ranked
        ]
        return {"params": params, "pages_scanned": len(pages), "rowcount": len(rows), "rows": rows}

    def finewiki_outbound_links(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"topic": _text(query, "topic"), "limit": _int(query, "limit", 40, 1, 100)}
        rows = []
        for page in self._topic_pages(params["topic"], params["limit"]):
            links = outbound_links(page.get("wikitext"))
            rows.append({"page_id": page["page_id"], "title": page["title"], "url": page.get("url"),
                         "outbound_link_count": len(links), "outbound_links": links[:25]})
        return {"params": params, "rowcount": len(rows), "rows": rows}

    # -- TPC-H report API ----------------------------------------------------

    def tpch_shipmode_priority(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"shipmode": _text(query, "shipmode"), "shipmode2": _text(query, "shipmode2"),
                  "year": _int(query, "year")}
        modes = {params["shipmode"].rstrip(), params["shipmode2"].rstrip()}
        rows = sorted(
            ({k: v for k, v in row.items() if k != "year"} for row in self._tpch["q12"]
             if row["year"] == params["year"] and row["l_shipmode"] in modes),
            key=lambda row: row["l_shipmode"],
        )
        return {"report": "TPC-H Q12: shipping modes and order priority", "params": params,
                "rowcount": len(rows), "rows": rows}

    def tpch_promo_revenue(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"year": _int(query, "year")}
        found = [row["promo_revenue"] for row in self._tpch["q14"] if row["year"] == params["year"]]
        rows = [{"promo_revenue": found[0] if found else None}]
        return {"report": "TPC-H Q14: promotion effect (September)", "params": params, "rowcount": 1, "rows": rows}

    def tpch_segment_revenue(self, query: Mapping[str, str]) -> Dict[str, Any]:
        params = {"segment": _text(query, "segment"), "year": _int(query, "year")}
        found = [row for row in self._tpch["segment"]
                 if row["year"] == params["year"] and row["segment"] == params["segment"].rstrip()]
        empty = {"revenue": None, "line_cnt": 0, "high_pri_lines": None, "low_pri_lines": None}
        row = {key: found[0][key] for key in empty} if found else empty
        return {"report": "Revenue and order-priority mix of a market segment", "params": params,
                "rowcount": 1, "rows": [row]}


class _Handler(BaseHTTPRequestHandler):
    server_version = "HaloMockAPI/1.0"
    server: "_Server"

    def do_GET(self) -> None:  # noqa: N802 (http.server naming)
        url = urllib.parse.urlsplit(self.path)
        query = dict(urllib.parse.parse_qsl(url.query, keep_blank_values=True))
        try:
            payload = self.server.api.handle(url.path, query)
        except BadRequest as exc:
            self._send(400, {"error": str(exc)})
        except LookupError:
            self._send(404, {"error": f"unknown endpoint {url.path}", "endpoints": sorted(self.server.api.routes)})
        except Unavailable as exc:
            self._send(503, {"error": str(exc)})
        except Exception as exc:
            LOGGER.exception("request %s failed", self.path)
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})
        else:
            self._send(200, payload)

    def _send(self, status: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=_json_default).encode("utf-8")
        # The reason phrase carries the error, so a client exception names the cause.
        reason = None if status == 200 else str(payload.get("error", ""))
        if reason is not None:
            reason = " ".join(reason.split()).encode("ascii", "replace").decode("ascii")[:200]
        self.send_response(status, reason)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        if self.server.verbose:
            super().log_message(format, *args)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], api: MockAPI, verbose: bool) -> None:
        super().__init__(address, _Handler)
        self.api = api
        self.verbose = verbose


def _connect_summary(executor: halo_db.PostgresDatabaseExecutor) -> str:
    kwargs = executor._effective_connect_kwargs()
    shown = {key: kwargs[key] for key in ("host", "port", "user", "dbname") if key in kwargs}
    return " ".join(f"{key}={value}" for key, value in shown.items()) or "libpq defaults"


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local mock API for the templates in templates/exps.")
    parser.add_argument("--host", default="127.0.0.1", help="listen address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="listen port; the templates use 8765")
    parser.add_argument("--datasets", default=",".join(DATASETS),
                        help="comma-separated endpoint groups to serve (default: imdb,finewiki,tpch)")
    parser.add_argument("--pg-host", help="Postgres host (default: HALO_PG_HOST or libpq default)")
    parser.add_argument("--pg-port", type=int, help="Postgres port (default: HALO_PG_PORT or 5432)")
    parser.add_argument("--pg-user", help="Postgres user (default: HALO_PG_USER or libpq default); "
                                          "the password comes from HALO_PG_PASSWORD, PGPASSWORD, or ~/.pgpass")
    parser.add_argument("--imdb-db", default="imdb", help="IMDb database name (default: imdb)")
    parser.add_argument("--finewiki-db", default="finewiki", help="FineWiki database name (default: finewiki)")
    parser.add_argument("--tpch-db", default="tpch", help="TPC-H database name (default: tpch)")
    parser.add_argument("--pool-size", type=int, default=8, help="connections kept per database (default: 8)")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[mock_api] %(message)s")

    datasets = [name.strip() for name in args.datasets.split(",") if name.strip()]
    unknown = sorted(set(datasets) - set(DATASETS))
    if unknown or not datasets:
        parser.error(f"--datasets takes a subset of {','.join(DATASETS)} (got {args.datasets!r})")
    if halo_db.psycopg is None:
        print("mock_api: psycopg is not installed; run `uv sync --extra postgres` first.", file=sys.stderr)
        return 2

    common = {key: value for key, value in (("host", args.pg_host), ("port", args.pg_port), ("user", args.pg_user))
              if value is not None}
    dbnames = {"imdb": args.imdb_db, "finewiki": args.finewiki_db, "tpch": args.tpch_db}
    executors = {
        name: halo_db.PostgresDatabaseExecutor(connect_kwargs={**common, "dbname": dbnames[name]},
                                               pool_size=args.pool_size, prepare_threshold=None)
        for name in datasets
    }
    api = MockAPI(executors)
    for name in datasets:
        try:
            api.check(name)
            if name == "tpch":
                LOGGER.info("tpch: aggregating the TPC-H reports once (minutes at scale factor 10)...")
                api.prepare_tpch()
        except Exception as exc:
            print(
                f"mock_api: cannot serve the {name} endpoints from Postgres ({_connect_summary(executors[name])}): "
                f"{' '.join(str(exc).split())}\n"
                "Load the data first (templates/exps/README.md), point --pg-host/--pg-port/--pg-user and "
                f"--{name}-db at it, or serve a subset with --datasets.",
                file=sys.stderr,
            )
            return 2
        LOGGER.info("%s: ready (%s)", name, _connect_summary(executors[name]))

    server = _Server((args.host, args.port), api, verbose=args.verbose)
    LOGGER.info("listening on http://%s:%d (%s)", args.host, server.server_port, ", ".join(datasets))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
