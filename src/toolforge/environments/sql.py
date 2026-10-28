"""SQL environment: a small deterministic SQLite database, read-only.

The schema is fixed in code (``SCHEMA``); fixtures supply rows. State shape::

    {"current_date": "2026-03-02", "tables": {"customers": [rows], "orders": [rows], "products": [rows]}}

Safety model (all verified by tests):

- The SQLite connection is built in-memory at reset and guarded by an
  authorizer that allows only SELECT/READ/FUNCTION/RECURSIVE, plus
  INSERT/UPDATE/DELETE against the *ephemeral* database (``""``) which SQLite
  uses internally for UNION de-duplication and sorting. Everything else —
  writes, DDL, ATTACH, PRAGMA — is denied and surfaced as an
  ``unsafe_operation`` failure.
- ``sqlite3`` itself rejects multi-statement strings.
- A progress handler aborts runaway queries (e.g. unbounded recursive CTEs)
  after ``progress_limit`` VM steps -> ``limit_exceeded`` failure.
- Results are capped at ``max_rows`` with a ``truncated`` flag.
"""

from __future__ import annotations

import random
import sqlite3
from typing import Any, ClassVar

from toolforge.environments.base import (
    ERR_INVALID_ARGUMENT,
    ERR_LIMIT,
    ERR_NOT_FOUND,
    ERR_UNSAFE,
    ERR_UNSUPPORTED,
    Environment,
    ToolResult,
    require,
)
from toolforge.environments.pools import PEOPLE, pick
from toolforge.schemas import ToolSpec
from toolforge.timeutils import DEFAULT_CURRENT_DATE, add_days

# Fixed deterministic schema: table -> (column, sqlite type) pairs.
SCHEMA: dict[str, list[tuple[str, str]]] = {
    "customers": [
        ("customer_id", "TEXT"), ("name", "TEXT"), ("region", "TEXT"),
        ("tier", "TEXT"), ("signup_date", "TEXT"),
    ],
    "products": [
        ("product_id", "TEXT"), ("name", "TEXT"), ("category", "TEXT"),
        ("price", "REAL"), ("stock", "INTEGER"),
    ],
    "orders": [
        ("order_id", "TEXT"), ("customer_id", "TEXT"), ("product_id", "TEXT"),
        ("quantity", "INTEGER"), ("unit_price", "REAL"), ("status", "TEXT"),
        ("order_date", "TEXT"),
    ],
}

REGIONS = ["north", "south", "east", "west"]
TIERS = ["basic", "pro", "enterprise"]
CATEGORIES = ["hardware", "software", "services"]
ORDER_STATUSES = ["pending", "shipped", "completed", "cancelled"]

_PRODUCT_SEED = [
    ("prd_001", "Sensor Kit A", "hardware", 129.0),
    ("prd_002", "Relay Module", "hardware", 45.5),
    ("prd_003", "Analytics Suite License", "software", 899.0),
    ("prd_004", "Support Plan", "services", 240.0),
    ("prd_005", "Gateway Device", "hardware", 310.0),
    ("prd_006", "Integration Toolkit", "software", 150.0),
]

# Authorizer policy
_ALLOWED_ACTIONS = frozenset({
    sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION, getattr(sqlite3, "SQLITE_RECURSIVE", 1 << 30),
})
_EPHEMERAL_WRITE_ACTIONS = frozenset({sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE})


def _row_matches_schema(table: str, row: Any, index: int) -> None:
    require(isinstance(row, dict), f"{table}[{index}] must be an object")
    columns = dict(SCHEMA[table])
    missing = set(columns) - set(row)
    require(not missing, f"{table}[{index}] missing column(s) {sorted(missing)}")
    extra = set(row) - set(columns)
    require(not extra, f"{table}[{index}] has unknown column(s) {sorted(extra)}")
    for col, sql_type in columns.items():
        value = row[col]
        if sql_type == "TEXT":
            require(isinstance(value, str), f"{table}[{index}].{col} must be a string")
        elif sql_type == "INTEGER":
            require(isinstance(value, int) and not isinstance(value, bool),
                    f"{table}[{index}].{col} must be an integer")
        elif sql_type == "REAL":
            require(isinstance(value, (int, float)) and not isinstance(value, bool),
                    f"{table}[{index}].{col} must be a number")


class SqlEnvironment(Environment):
    family: ClassVar[str] = "sql"
    volatile_keys: ClassVar[frozenset[str]] = frozenset()  # queries never mutate state

    def __init__(self, fixture: dict[str, Any] | None = None, seed: int = 0,
                 max_rows: int = 50, progress_limit: int = 1_000_000) -> None:
        self._max_rows = max_rows
        self._progress_limit = progress_limit
        self._conn: sqlite3.Connection | None = None
        self._progress_ticks = 0
        super().__init__(fixture=fixture, seed=seed)

    # ------------------------------------------------------------------ state

    def default_state(self, seed: int) -> dict[str, Any]:
        rng = random.Random(seed)
        customers = []
        for i, (name, _) in enumerate(PEOPLE):
            customers.append({
                "customer_id": f"cus_{i + 1:03d}",
                "name": name,
                "region": REGIONS[(i + rng.randrange(2)) % len(REGIONS)],
                "tier": TIERS[(i * 2 + rng.randrange(2)) % len(TIERS)],
                "signup_date": add_days(DEFAULT_CURRENT_DATE, -(30 + i * 45 + rng.randrange(20))),
            })
        # Two extra org-like customers for variety.
        for j, org in enumerate(["Meridian Labs", "Atlas Freight"]):
            customers.append({
                "customer_id": f"cus_{len(PEOPLE) + j + 1:03d}",
                "name": org,
                "region": pick(rng, REGIONS),
                "tier": "enterprise",
                "signup_date": add_days(DEFAULT_CURRENT_DATE, -(60 + j * 30)),
            })
        products = [
            {"product_id": pid, "name": name, "category": cat, "price": price,
             "stock": 5 + rng.randrange(95)}
            for pid, name, cat, price in _PRODUCT_SEED
        ]
        # One product deterministically low-stock for threshold queries.
        products[1]["stock"] = 4
        orders = []
        for i in range(20):
            product = products[rng.randrange(len(products))]
            customer = customers[rng.randrange(len(customers))]
            status = ORDER_STATUSES[i % 10 if i % 10 < 4 else rng.randrange(4)]
            orders.append({
                "order_id": f"ord_{i + 1:03d}",
                "customer_id": customer["customer_id"],
                "product_id": product["product_id"],
                "quantity": 1 + rng.randrange(6),
                "unit_price": product["price"],
                "status": status,
                "order_date": add_days(DEFAULT_CURRENT_DATE, -(1 + (i * 7 + rng.randrange(5)) % 28)),
            })
        return {"current_date": DEFAULT_CURRENT_DATE,
                "tables": {"customers": customers, "products": products, "orders": orders}}

    def normalize_state(self, fixture: dict[str, Any]) -> dict[str, Any]:
        require(isinstance(fixture, dict), "sql fixture must be an object")
        raw_tables = fixture.get("tables")
        require(isinstance(raw_tables, dict), "fixture.tables must be an object keyed by table name")
        unknown = set(raw_tables) - set(SCHEMA)
        require(not unknown, f"unknown table(s) in fixture: {sorted(unknown)}")
        tables: dict[str, list[dict]] = {}
        for table in SCHEMA:
            rows = raw_tables.get(table, [])
            require(isinstance(rows, list), f"tables.{table} must be a list of row objects")
            for i, row in enumerate(rows):
                _row_matches_schema(table, row, i)
            tables[table] = [dict(row) for row in rows]
        return {"current_date": str(fixture.get("current_date", DEFAULT_CURRENT_DATE)),
                "tables": tables}

    def _post_reset(self) -> None:
        """(Re)build the in-memory SQLite database from state rows."""
        if self._conn is not None:
            self._conn.close()
        self._conn = sqlite3.connect(":memory:")
        for table, columns in SCHEMA.items():
            cols_sql = ", ".join(f"{name} {sql_type}" for name, sql_type in columns)
            self._conn.execute(f"CREATE TABLE {table} ({cols_sql})")
            rows = self._state["tables"].get(table, [])
            if rows:
                names = [name for name, _ in columns]
                placeholders = ", ".join("?" * len(names))
                self._conn.executemany(
                    f"INSERT INTO {table} ({', '.join(names)}) VALUES ({placeholders})",
                    [tuple(row[name] for name in names) for row in rows],
                )
        # Authorizer and progress handler are attached only after data is loaded.
        self._conn.set_authorizer(self._authorize)
        self._conn.set_progress_handler(self._progress, 1)

    def _authorize(self, action: int, arg1: Any, arg2: Any, db_name: Any, trigger: Any) -> int:
        if action in _ALLOWED_ACTIONS:
            return sqlite3.SQLITE_OK
        if action in _EPHEMERAL_WRITE_ACTIONS and db_name == "":
            # SQLite-internal ephemeral tables (UNION dedup, sorting). Not
            # reachable by name from user SQL.
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    def _progress(self) -> int:
        self._progress_ticks += 1
        return 1 if self._progress_ticks > self._progress_limit else 0

    # ------------------------------------------------------------------ tools

    tool_specs: ClassVar[list[ToolSpec]] = [
        ToolSpec(
            "sql.list_tables",
            "List the tables in the database with their row counts and columns.",
            {"type": "object", "properties": {}, "additionalProperties": False},
        ),
        ToolSpec(
            "sql.describe_table",
            "Describe one table: column names/types, row count, and a sample row.",
            {"type": "object", "properties": {"table": {"type": "string", "minLength": 1}},
             "required": ["table"], "additionalProperties": False},
        ),
        ToolSpec(
            "sql.run_query",
            "Run a single read-only SELECT query and return columns plus rows. "
            "Writes, DDL, ATTACH and PRAGMA are rejected. Use '?' placeholders "
            "with the params list for literal values.",
            {"type": "object", "properties": {
                "sql": {"type": "string", "minLength": 1, "maxLength": 4000,
                        "description": "A single SELECT statement."},
                "params": {"type": "array", "maxItems": 20,
                           "items": {"type": ["string", "number", "boolean", "null"]},
                           "description": "Bind values for '?' placeholders."},
            }, "required": ["sql"], "additionalProperties": False},
        ),
    ]

    # ------------------------------------------------------------ tool bodies

    def _tool_list_tables(self, args: dict[str, Any]) -> ToolResult:
        return self._ok({
            "tables": [
                {"table": name, "row_count": len(self._state["tables"].get(name, [])),
                 "columns": [c for c, _ in cols]}
                for name, cols in SCHEMA.items()
            ]
        })

    def _tool_describe_table(self, args: dict[str, Any]) -> ToolResult:
        table = args["table"]
        if table not in SCHEMA:
            return self._fail(ERR_NOT_FOUND,
                              f"No table {table!r}; available: {sorted(SCHEMA)}")
        rows = self._state["tables"].get(table, [])
        return self._ok({
            "table": table,
            "columns": [{"name": n, "type": t} for n, t in SCHEMA[table]],
            "row_count": len(rows),
            "sample_row": rows[0] if rows else None,
        })

    def _tool_run_query(self, args: dict[str, Any]) -> ToolResult:
        sql = args["sql"].strip()
        params = list(args.get("params") or [])
        if not sql.lower().lstrip().startswith(("select", "with")):
            return self._fail(ERR_UNSAFE,
                              "Only read-only SELECT queries are allowed "
                              "(statements may start with SELECT or WITH).")
        assert self._conn is not None  # guaranteed by _post_reset
        self._progress_ticks = 0
        try:
            cursor = self._conn.execute(sql, params)
        except sqlite3.Error as exc:
            message = str(exc)
            if "interrupted" in message:
                return self._fail(ERR_LIMIT,
                                  f"Query aborted: exceeded {self._progress_limit} execution "
                                  "steps (possible unbounded recursion).")
            if "not authorized" in message:
                return self._fail(ERR_UNSAFE,
                                  "Query rejected: writes, DDL, ATTACH and PRAGMA are not "
                                  "allowed in this read-only environment.")
            if "one statement" in message.lower():
                return self._fail(ERR_UNSUPPORTED,
                                  "Multiple statements are not supported; send exactly one "
                                  "SELECT query per call.")
            return self._fail(ERR_INVALID_ARGUMENT, f"Query failed: {message}")
        columns = [d[0] for d in cursor.description] if cursor.description else []
        rows = cursor.fetchmany(self._max_rows + 1)
        truncated = len(rows) > self._max_rows
        rows = rows[: self._max_rows]
        return self._ok({
            "columns": columns,
            "rows": [list(r) for r in rows],
            "row_count": len(rows),
            "truncated": truncated,
        })
