"""Evidence retention sweep (F2-E).

Deletes expired evidence from storage and the database, writing an audit row
for every deletion/failure. Storage is deleted before metadata so rows never
point at missing files; failed deletions keep the row for retry.

Usage:
  python scripts/run_retention_sweep.py            # run the sweep
  python scripts/run_retention_sweep.py --dry-run  # report only, delete nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Delete expired evidence")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="report what would be deleted without changing anything",
    )
    args = parser.parse_args()

    from packages.config import Settings
    from packages.db import base as db_base
    from services.evidence.retention import run_retention_sweep
    from services.evidence.store import LocalDiskEvidenceStore

    settings = Settings()
    db_base.configure(settings.database_url)

    store = LocalDiskEvidenceStore(settings.evidence.root)
    with db_base.session_scope() as session:
        report = run_retention_sweep(
            session,
            store,
            allow_active_event_deletion=settings.evidence.allow_active_event_deletion,
            dry_run=args.dry_run,
        )

    print(json.dumps(report.as_dict(), indent=2))
    return 1 if report.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
