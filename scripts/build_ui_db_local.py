from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.ui import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH, materialize_ui_database


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Scam Intelligence Workbench SQLite database from a local experiment config.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--model", default="auto")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    db_path = materialize_ui_database(args.config, args.db, model_name=args.model, seed=args.seed)
    print(db_path)


if __name__ == "__main__":
    main()
