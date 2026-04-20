from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from scam_gnn.config import load_config
from scam_gnn.data.public_datasets import prepare_phiusiil_url_dataset, prepare_uci_sms_spam_dataset
from scam_gnn.data.synthetic import generate_synthetic_multichannel_events
from scam_gnn.pipeline import run_experiment_from_config
from scam_gnn.ui import DEFAULT_CONFIG_PATH, DEFAULT_DB_PATH, materialize_ui_database, serve_ui


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Research scaffold for relationship-aware scam GNN experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    synthetic_parser = subparsers.add_parser("generate-synthetic", help="Generate a synthetic multi-channel dataset.")
    synthetic_parser.add_argument("--output", required=True, help="CSV path for the generated events table.")
    synthetic_parser.add_argument("--seed", type=int, default=7)
    synthetic_parser.add_argument("--num-events", type=int, default=96)

    uci_parser = subparsers.add_parser("prepare-uci-sms", help="Download and standardize the official UCI SMS Spam Collection.")
    uci_parser.add_argument("--output", default="data/public/uci_sms_spam_collection.csv")
    uci_parser.add_argument("--force", action="store_true")

    phiusiil_parser = subparsers.add_parser("prepare-phiusiil-url", help="Download and standardize the official UCI PhiUSIIL phishing URL dataset.")
    phiusiil_parser.add_argument("--output", default="data/public/phiusiil_url_collection.csv")
    phiusiil_parser.add_argument("--force", action="store_true")

    inspect_parser = subparsers.add_parser("inspect-config", help="Print a config file back as JSON.")
    inspect_parser.add_argument("--config", required=True)

    run_parser = subparsers.add_parser("run-research", help="Run the configured experiment.")
    run_parser.add_argument("--config", required=True)

    ui_db_parser = subparsers.add_parser("build-ui-db", help="Materialize a SQLite database for the analyst UI.")
    ui_db_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ui_db_parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    ui_db_parser.add_argument("--model", default="auto")
    ui_db_parser.add_argument("--seed", type=int, default=None)

    serve_parser = subparsers.add_parser("serve-ui", help="Serve the analyst UI and JSON endpoints.")
    serve_parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    serve_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    serve_parser.add_argument("--model", default="auto")
    serve_parser.add_argument("--seed", type=int, default=None)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--rebuild", action="store_true")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.command == "generate-synthetic":
        frame = generate_synthetic_multichannel_events(seed=args.seed, num_events=args.num_events)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output_path, index=False)
        print(output_path)
        return
    if args.command == "prepare-uci-sms":
        print(prepare_uci_sms_spam_dataset(args.output, force=args.force))
        return
    if args.command == "prepare-phiusiil-url":
        print(prepare_phiusiil_url_dataset(args.output, force=args.force))
        return
    if args.command == "inspect-config":
        config = load_config(args.config)
        print(json.dumps(asdict(config), indent=2))
        return
    if args.command == "run-research":
        summary = run_experiment_from_config(args.config)
        print(json.dumps(summary, indent=2, default=str))
        return
    if args.command == "build-ui-db":
        print(materialize_ui_database(args.config, args.db, model_name=args.model, seed=args.seed))
        return
    if args.command == "serve-ui":
        db_path = Path(args.db)
        if args.rebuild or not db_path.exists():
            materialize_ui_database(args.config, db_path, model_name=args.model, seed=args.seed)
        serve_ui(db_path, host=args.host, port=args.port)
        return


if __name__ == "__main__":
    main()
