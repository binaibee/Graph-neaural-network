from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.ui import DEFAULT_DB_PATH, serve_ui


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Scam Intelligence Workbench from a local SQLite database.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    serve_ui(Path(args.db), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
