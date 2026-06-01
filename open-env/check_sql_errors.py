"""Surface the SQL INSERT errors that AWM env reset swallows.

When the AWM env resets, it seeds a SQLite DB via create_database(), whose
_insert_sample_data() runs each INSERT in a try/except that only logs a warning
("Failed to insert into ...") and keeps going. So reset reports success even
when sample rows silently fail to load.

This script reproduces that exact seeding path in-process for each scenario,
captures the warnings the db_manager logger emits, and reports the ratio of
failed vs successful INSERTs.

    uv run python open-env/check_sql_errors.py [--scenario NAME] [--limit N]
"""

import argparse
import logging
import tempfile

from agent_world_model_env.server import db_manager
from agent_world_model_env.server.data_loader import AWMDataLoader


def count_insert_statements(sample_data) -> int:
    """Total INSERTs _insert_sample_data would attempt for this sample_data."""
    if not sample_data:
        return 0
    if isinstance(sample_data, dict) and "tables" in sample_data:
        sample_data = sample_data["tables"]
    if not isinstance(sample_data, list):
        return 0
    total = 0
    for item in sample_data:
        if isinstance(item, dict):
            total += len([s for s in item.get("insert_statements", []) if str(s).strip()])
        elif isinstance(item, str) and item.strip():
            total += 1
    return total


def count_schema_statements(db_schema) -> tuple[int, int]:
    """Total DDL and index statements _create_schema would attempt."""
    ddl = 0
    index = 0
    for table in db_schema.get("tables", []):
        if table.get("ddl", "").strip():
            ddl += 1
        index += len([i for i in table.get("indexes", []) if str(i).strip()])
    return ddl, index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", help="Scan only this scenario.")
    parser.add_argument("--limit", type=int, help="Scan only the first N scenarios.")
    args = parser.parse_args()

    # Capture the warnings db_manager swallows.
    records: list[logging.LogRecord] = []

    class CaptureHandler(logging.Handler):
        def emit(self, record):
            records.append(record)

    db_manager.logger.addHandler(CaptureHandler())
    db_manager.logger.setLevel(logging.WARNING)

    loader = AWMDataLoader()
    scenarios = loader.list_scenarios()
    if args.scenario:
        scenarios = [s for s in scenarios if s["name"] == args.scenario]
        if not scenarios:
            raise SystemExit(f"Scenario '{args.scenario}' not found")
    if args.limit:
        scenarios = scenarios[: args.limit]

    # [attempted, failed] per statement kind.
    totals = {"insert": [0, 0], "ddl": [0, 0], "index": [0, 0]}
    scenarios_with_failures = 0

    with tempfile.TemporaryDirectory() as tmp:
        for s in scenarios:
            name = s["name"]
            db_schema = loader.get_db_schema(name)
            sample_data = loader.get_sample_data(name)
            n_insert = count_insert_statements(sample_data)
            n_ddl, n_index = count_schema_statements(db_schema)

            records.clear()
            db_manager.create_database(f"{tmp}/{name}.db", db_schema, sample_data)

            insert_errs = [r for r in records if r.getMessage().startswith("Failed to insert into")]
            ddl_errs = [r for r in records if r.getMessage().startswith("Failed to execute DDL")]
            index_errs = [r for r in records if r.getMessage().startswith("Failed to create index")]

            totals["insert"][0] += n_insert
            totals["insert"][1] += len(insert_errs)
            totals["ddl"][0] += n_ddl
            totals["ddl"][1] += len(ddl_errs)
            totals["index"][0] += n_index
            totals["index"][1] += len(index_errs)

            if insert_errs or ddl_errs or index_errs:
                scenarios_with_failures += 1
                ratio = len(insert_errs) / n_insert if n_insert else 0.0
                print(
                    f"{name}: inserts {n_insert - len(insert_errs)}/{n_insert} ok, "
                    f"{len(insert_errs)} failed (ratio {ratio:.3f}), "
                    f"ddl {len(ddl_errs)}/{n_ddl} failed, index {len(index_errs)}/{n_index} failed"
                )
                for r in insert_errs + ddl_errs + index_errs:
                    print(f"    {r.getMessage()}")

    print("=" * 80)
    print(f"scenarios scanned:        {len(scenarios)}")
    print(f"scenarios with failures:  {scenarios_with_failures}")
    for kind in ("insert", "ddl", "index"):
        attempted, failed = totals[kind]
        ratio = failed / attempted if attempted else 0.0
        print(
            f"{kind:8}  attempted {attempted:6}  failed {failed:5}  normal {attempted - failed:6}  "
            f"error ratio {ratio:.4f} ({ratio * 100:.2f}%)"
        )
    all_attempted = sum(v[0] for v in totals.values())
    all_failed = sum(v[1] for v in totals.values())
    if all_attempted:
        ratio = all_failed / all_attempted
        print(
            f"{'overall':8}  attempted {all_attempted:6}  failed {all_failed:5}  "
            f"normal {all_attempted - all_failed:6}  error ratio {ratio:.4f} ({ratio * 100:.2f}%)"
        )


if __name__ == "__main__":
    main()
