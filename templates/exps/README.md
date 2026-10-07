# Evaluation workloads W1–W6

These are the six workload templates of the paper's evaluation (Fig. 5), prepared
for this prototype. Their DAG topology, models, SQL statements, rule-processor
configurations, and per-operator latency means are the ones evaluated. What changed
is that each HTTP operator now issues a real request to a local mock API and passes
the JSON answer to the agent that consumes it; Halo still pads every call to a
Gamma-distributed latency with the evaluated mean (3 s or 6 s).

| W | File | Dataset (Postgres DB) | Fig. 5 title | LLM / tool operators | Tools |
|---|---|---|---|---|---|
| W1 | `w1_imdb_diamond.yaml` | IMDb (`imdb`) | IMDb-Diamond | 8 / 9 | 7 SQL, 2 HTTP |
| W2 | `w2_imdb_triple_chain.yaml` | IMDb (`imdb`) | IMDb-TripleChain | 10 / 3 | 2 SQL, 1 HTTP |
| W3 | `w3_finewiki_long_chain.yaml` | FineWiki (`finewiki`) | FineWiki-LongChain | 9 / 6 | 4 SQL, 2 HTTP |
| W4 | `w4_finewiki_bridge.yaml` | FineWiki (`finewiki`) | FineWiki-Bridge | 9 / 3 | 2 SQL, 1 HTTP |
| W5 | `w5_tpch_trident.yaml` | TPC-H SF 10 (`tpch`) | TPCH-Trident | 7 / 10 | 7 SQL, 2 HTTP, 1 rule processor |
| W6 | `w6_tpch_fanout.yaml` | TPC-H SF 10 (`tpch`) | TPCH-Fanout | 9 / 13 | 10 SQL, 2 HTTP, 1 rule processor |

Tool operators are counted after atomization: every `db_queries` entry becomes its
own SQL operator, and HTTP and rule-processor nodes count once each. LLM nodes use
`Qwen/Qwen3-14B`, `openai/gpt-oss-20b`, or `Qwen/Qwen3-32B`, as shaded in Fig. 5.

## HTTP operators

Each HTTP node keeps the id of the evaluated template and serves the lookup that
the SQL it replaced computed. The mock API (`scripts/mock_api.py`) answers from the
same databases, so the payload is real data:

| W | Node | Request | Payload rows | Mean latency |
|---|---|---|---|---|
| W1 | `http_entry_movie_by_title` | `GET /imdb/titles/search?keyword=<search_keyword>&limit=25` | Movies, TV movies, and TV series whose primary or original title contains the keyword, most-voted first: `tconst`, `primary_title`, `original_title`, `start_year`, `title_type`, `average_rating`, `num_votes` | 3 s |
| W1 | `http_entry_movie_recent` | `GET /imdb/titles/recent?keyword=<search_keyword>&since=2000&limit=15` | Movies and TV movies released in 2000 or later that match the keyword, most-voted first: `tconst`, `primary_title`, `start_year`, `genres`, `average_rating`, `num_votes` | 3 s |
| W2 | `http_people_overview` | `GET /imdb/people/search?keyword=<search_keyword>&limit=40` | People whose name contains the keyword, in name order: `nconst`, `primary_name`, `primary_profession`, `known_for_titles` | 3 s |
| W3 | `http_fw_category_pages` | `GET /finewiki/categories?topic=<user_query>&limit=30` | Wikipedia categories of the articles whose title contains the topic, most shared first: `category`, `title`, `url`, `member_count`, `member_pages` | 3 s |
| W3 | `http_fw_outbound_links` | `GET /finewiki/outbound-links?topic=<user_query>&limit=40` | Articles whose title contains the topic, each with the article titles it links to: `page_id`, `title`, `url`, `outbound_link_count`, `outbound_links` (first 25) | 3 s |
| W4 | `http_fw_category_pages` | `GET /finewiki/categories?topic=<user_query>&limit=20` | As in W3 | 3 s |
| W5 | `http_tpch_q12_shipmode` | `GET /tpch/reports/shipmode-priority?shipmode=..&shipmode2=..&year=..` | TPC-H Q12, one row per ship mode: `l_shipmode`, `line_count`, and the lines of high-priority (`1-URGENT`, `2-HIGH`) and other orders, `high_line_count` and `low_line_count`, among lines received that year after their commit date and shipped before it | 6 s |
| W5 | `http_tpch_q14_promo` | `GET /tpch/reports/promo-revenue?year=..` | TPC-H Q14: `promo_revenue`, the percentage of September revenue from `PROMO%` parts | 6 s |
| W6 | `http_s1_q3_q3_segment_revenue` | `GET /tpch/reports/segment-revenue?segment=..&year=..` | For the segment's orders placed that year: `revenue`, `line_cnt`, `high_pri_lines`, `low_pri_lines`, over lines shipped after the order date | 6 s |
| W6 | `http_s1_q3_q12_shipmode_priority` | `GET /tpch/reports/shipmode-priority?shipmode=..&shipmode2=..&year=..` | As W5's Q12 report | 6 s |

The TPC-H parameters come from `tpch_params`. Every response is a JSON object with
`endpoint`, `params`, `rowcount`, and `rows` (plus `report` for TPC-H), so Halo
renders it into the consuming agent's prompt like a SQL result, and each prompt
describes the payload it receives. The IMDb and FineWiki endpoints run the replaced
SQL with a unique tiebreaker added to `ORDER BY`, so answers are deterministic. The
TPC-H reports are aggregated once when the service starts, over every year, ship
mode, and segment; each request then returns the numbers its replaced query computes,
without another scan of `lineitem`. The replaced category query matched titles
starting with `Category:`, which FineWiki (articles only) never contains, so the
category endpoint reads the `[[Category:...]]` tags of the articles the link
endpoint retrieves.

## Rule processor

`rule_param_extractor` (W5 and W6, the orange diamond in Fig. 5) runs the local
function `regex_param_extractor` instead of an LLM call. It applies per-field regular
expressions to `user_query` (the first matching pattern wins), fills unmatched fields
with defaults, and normalizes the values: upper case, enum canonicalization (e.g.
`MIDDLE` becomes `MIDDLE EAST`), ISO dates, and numbers. The resulting `tpch_params`
dict binds every SQL statement and API request of the template. W5 extracts date,
year, region, segment, brand, part type and size, two ship modes, a discount band,
and a quantity; W6 extracts year, segment, brand, two ship modes, a discount band,
and a quantity, and keeps region at its default. The patterns and defaults are the
evaluated ones; for example, W6 reads `qty=24` but not `qty<24`, so such inputs keep
the default quantity.

## Latency model

An HTTP node runs in one of three modes, chosen by the keys it declares:

- `url` with `latency_s` or `latency_ms` (these templates): Halo issues the request,
  then pads the call to a latency sampled from a Gamma distribution with that mean,
  sleeping only for the remainder. An answer slower than its sampled latency is not
  cut short, so the paper's latency model holds while the service answers within it.
  The bundled mock API caches every answer.
- `sleep_s` or `sleep_ms`: latency injection alone; a `url` is never contacted. This
  is how the paper's runs served HTTP calls; replace `latency_s` with `sleep_s` in a
  template to reproduce them without the mock API.
- `url` alone: a live request, with `timeout_s` or `timeout_ms` as its timeout.

Before planning, Halo profiles HTTP nodes on the sample contexts. A request whose
inputs are produced upstream (such as `search_keyword` or `tpch_params`) cannot be
bound then, so it is not sent; the call still takes a sampled latency, so the
planner's cost for the node stays around the declared mean.

## Setup

### 1. Load the databases

This prototype does not ship data loaders. The templates expect three Postgres
databases, `imdb`, `finewiki`, and `tpch`, on one server.

- IMDb: download `title.basics`, `title.ratings`, `name.basics`, `title.crew`, and
  `title.akas` from <https://datasets.imdbws.com/> into tables of the same names in
  snake case (`title_basics`, ...), with snake-case columns (`tconst`,
  `primary_title`, `start_year`, `num_votes`, `primary_name`, `known_for_titles`,
  `title_id`, `is_original_title`, ...). The files are tab-separated with `\N` for
  NULL; drop rows with a wrong column count before `COPY`. The evaluation also
  indexed `title_basics(start_year)`, `title_ratings(num_votes)`,
  `name_basics(primary_name)`, and `title_akas(region)`.
- FineWiki: create a table `pages` with the columns of `HuggingFaceFW/finewiki`
  (`id`, `wikiname`, `page_id` as primary key, `title`, `url`, `date_modified`,
  `in_language`, `wikidata_id`, `bytes_html`, `wikitext`, `version`, `infoboxes`,
  `has_math`) and insert 20,000 rows of its `en` subset (the paper's size), e.g.
  the first 20,000 of `datasets.load_dataset("HuggingFaceFW/finewiki", "en",
  split="train", streaming=True)`. The evaluation inputs are article titles sampled
  from the first 2,048 rows of that stream, so they are in the table.
- TPC-H: generate scale factor 10 with `dbgen` (<https://github.com/electrum/tpch-dbgen>),
  create the standard schema, strip the trailing `|` of every `.tbl` line, load each
  file with `COPY ... WITH (FORMAT csv, DELIMITER '|')`, and `ANALYZE`.

### 2. Start the mock API

```bash
uv sync --extra postgres
export HALO_PG_PASSWORD=...          # if the server needs one
python scripts/mock_api.py --pg-host localhost --pg-port 5432 --pg-user postgres
```

The service listens on `http://127.0.0.1:8765`, the URL in the templates, and logs
`listening on ...` once ready. At startup it checks each database and exits with an
error naming the database if one is unreachable or lacks its tables; it then
aggregates the TPC-H reports, which takes minutes at scale factor 10. Pass
`--datasets imdb` (or `finewiki`, `tpch`) to serve a subset, and `--imdb-db`,
`--finewiki-db`, `--tpch-db` for other database names. If an HTTP proxy is
configured, also set `no_proxy=127.0.0.1,localhost` for the Halo process. The
endpoints read the same server as the templates' SQL operators, so uncached answers
add to that server's load; IMDb keyword searches scan `title_basics` or
`name_basics`.

### 3. Run a template

```python
from halo import GraphOptimizer, GraphTemplateParser, MultiProcessGraphProcessor
from halo.query_planner import QueryPlanEvaluator

pg = {"dbname": "tpch", "user": "postgres", "host": "localhost", "port": 5432}
graph = GraphTemplateParser("templates/exps/w5_tpch_trident.yaml").parse()
queries = ["region=AMERICA segment=HOUSEHOLD year=1993 shipmode=MAIL", "brand=Brand#16 year=1997"]
contexts = [{"user_query": q} for q in queries]

optimizer = GraphOptimizer(num_gpus=4, num_cpu_workers=2, scheduler_mode="dp", plan_mode="profiled",
                           plan_evaluator=QueryPlanEvaluator(explainer_connect_kwargs=pg))
plan = optimizer.build_plan(graph, sample_contexts=contexts[:4], input_query_count=len(contexts))

processor = MultiProcessGraphProcessor(db_connect_kwargs=pg, max_batch_size="auto", persistent_workers=True)
try:
    results = processor.run_batch(plan, graph, contexts)
finally:
    processor.close()
print(results[0]["final_answer"])
```

Use `dbname` `imdb` for W1/W2 and `finewiki` for W3/W4. Inputs follow the
evaluation: natural-language movie questions for W1/W2 (e.g. "Who directed
Inception?"), FineWiki article titles for W3/W4 (e.g. "1874–75 FA Cup"), and TPC-H
parameter strings for W5/W6. The GPU workers must fit the 32B model.

## Differences from the evaluated templates

- HTTP nodes declare `url`, `method`, `params`, and `latency_s` instead of `sleep_s`
  alone, with the same means. W6's two HTTP nodes take only `tpch_params`, since the
  request does not use `user_query`.
- Graph names and descriptions identify the workload; stale comments are gone; W6's
  query names drop the `_light_v3` suffix.
- Prompts describe the inputs they receive, including each API payload.
- W2 keeps a direct edge from `keyword_planner` to `final_answer_qwen14b`, which
  Fig. 5 does not draw.
