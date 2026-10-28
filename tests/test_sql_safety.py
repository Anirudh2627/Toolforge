"""SQL safety tests: read-only enforcement, statement limits, runaway queries.

These are adversarial by design — they simulate what a misbehaving or
reward-hacking policy might emit.
"""

from __future__ import annotations

import pytest

from toolforge.environments import ERR_LIMIT, ERR_UNSAFE, ERR_UNSUPPORTED
from toolforge.environments.sql import SqlEnvironment


@pytest.fixture
def env():
    # Small progress limit keeps the runaway-query test fast.
    return SqlEnvironment(seed=31, progress_limit=200_000)


class TestWriteRejection:
    @pytest.mark.parametrize("sql", [
        "INSERT INTO orders VALUES ('ord_999','cus_001','prd_001',1,1.0,'pending','2026-03-01')",
        "UPDATE orders SET status = 'completed'",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "ALTER TABLE orders RENAME TO old_orders",
        "CREATE TABLE evil (x TEXT)",
        "CREATE INDEX idx ON orders(status)",
        "ATTACH DATABASE ':memory:' AS evil",
        "PRAGMA table_info(orders)",
        # Case/whitespace obfuscation must not bypass the authorizer.
        "  upDaTe orders set status='completed'",
        "SELECT 1; DROP TABLE orders",
        # Passes the SELECT/WITH prefix check; the authorizer must still deny.
        "WITH x AS (SELECT 1) INSERT INTO orders SELECT 'ord_999','cus_001','prd_001',1,1.0,'pending','2026-03-01' FROM x",
    ])
    def test_unsafe_statements_rejected(self, env, sql):
        res = env.execute("run_query", {"sql": sql})
        assert not res.ok
        assert res.error["code"] in (ERR_UNSAFE, ERR_UNSUPPORTED)

    def test_rejected_write_leaves_state_untouched(self, env):
        before = env.snapshot()
        res = env.execute("run_query", {"sql": "DELETE FROM orders"})
        assert not res.ok
        assert env.snapshot() == before
        assert env.execute("run_query", {"sql": "SELECT COUNT(*) FROM orders"}).output["rows"] == [[20]]

    def test_non_select_prefix_rejected_before_execution(self, env):
        res = env.execute("run_query", {"sql": "VACUUM"})
        assert not res.ok and res.error["code"] == ERR_UNSAFE


class TestStatementLimits:
    def test_multiple_statements_rejected(self, env):
        res = env.execute("run_query", {"sql": "SELECT 1; SELECT 2"})
        assert not res.ok
        assert res.error["code"] == ERR_UNSUPPORTED

    def test_runaway_recursive_cte_is_aborted(self, env):
        res = env.execute("run_query", {"sql": """
            WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r)
            SELECT count(*) FROM r
        """})
        assert not res.ok
        assert res.error["code"] == ERR_LIMIT
        assert "steps" in res.error["message"]

    def test_bounded_recursive_cte_works(self, env):
        res = env.execute("run_query", {"sql": """
            WITH RECURSIVE r(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM r WHERE n < 10)
            SELECT max(n) FROM r
        """})
        assert res.ok and res.output["rows"] == [[10]]

    def test_params_must_be_scalars(self, env):
        # Schema validation happens in the executor; the environment itself must
        # not crash on hostile params passed directly.
        res = env.execute("run_query", {"sql": "SELECT ?", "params": [["nested"]]})
        assert not res.ok  # sqlite3 rejects unsupported binding types


class TestReadPaths:
    def test_select_and_with_allowed(self, env):
        assert env.execute("run_query", {"sql": "SELECT status, COUNT(*) FROM orders GROUP BY status"}).ok
        assert env.execute("run_query",
                           {"sql": "WITH t AS (SELECT * FROM orders) SELECT COUNT(*) FROM t"}).ok

    def test_union_allowed(self, env):
        res = env.execute("run_query", {"sql": """
            SELECT customer_id FROM orders WHERE status='pending'
            UNION
            SELECT customer_id FROM orders WHERE status='shipped'
        """})
        assert res.ok

    def test_subquery_and_order_limit(self, env):
        res = env.execute("run_query", {"sql": """
            SELECT name FROM customers WHERE customer_id IN
              (SELECT customer_id FROM orders GROUP BY customer_id HAVING COUNT(*) > 2)
            ORDER BY name LIMIT 5
        """})
        assert res.ok and res.output["row_count"] <= 5

    def test_comment_obfuscated_write_rejected(self, env):
        res = env.execute("run_query",
                          {"sql": "SELECT 1 /* hi */; DELETE FROM orders", })
        assert not res.ok
