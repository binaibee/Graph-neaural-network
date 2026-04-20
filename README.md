# Relationship-Aware Scam Detection Using Graph Neural Networks Across Multi-Channel Communications

This repository is a research-oriented scaffold for building and evaluating relationship-aware scam detection models across SMS, email, and URL/domain infrastructure signals.

The codebase is designed around two tracks:

- `benchmark`: adapters for standardized public fraud/abuse benchmarks such as FDB.
- `custom`: a heterogeneous multi-channel graph pipeline that fuses public SMS, email, and phishing URL data into one graph-centric experiment.

## Current Status

The scaffold is implemented to run end-to-end on synthetic multi-channel data using only `numpy` and `pandas`.

Optional model paths are also implemented for:

- `scikit-learn` tree baselines
- `sentence-transformers` MiniLM embeddings
- `torch` and `torch-geometric` GNN models

Those optional dependencies are not installed in the current workspace, so the CLI and tests automatically fall back or skip the relevant steps.

## Core Design

Standardized tables:

- `events`: one row per communication artifact
- `entities`: canonical nodes for senders, recipients, addresses, URLs, and domains
- `relations`: typed, timestamped edges
- `splits`: rolling time-based train/validation/test assignments

Graph schema:

- Node types: `message`, `sender`, `recipient`, `phone`, `email_address`, `url`, `domain`
- Edge types: `sent_by`, `sent_to`, `uses_phone`, `uses_email`, `contains_url`, `resolves_to_domain`, `shares_sender`, `shares_recipient`, `shares_infrastructure`

Model ladder:

- Logistic regression baseline
- Tree ensemble baseline
- MLP baseline
- Homogeneous GCN and GraphSAGE
- Message-centric, relation-pruned heterogeneous GraphSAGE in PyG
- Message-centric heterogeneous HGT with `HGTConv`

## Quick Start

Install the minimal package:

```bash
python -m pip install -e .
```

If editable install is blocked by environment permissions, run the module directly with `PYTHONPATH=src`.

Generate synthetic multi-channel data and run the smoke experiment:

```bash
python -m scam_gnn.cli generate-synthetic --output artifacts/synthetic_events.csv
python -m scam_gnn.cli run-research --config examples/configs/synthetic_research.json
```

Windows PowerShell alternative without editable install:

```powershell
$env:PYTHONPATH='src'
python -m scam_gnn.cli run-research --config examples\configs\synthetic_research.json
```

Repo-root script alternatives:

```powershell
python scripts\build_ui_db_local.py --config examples\configs\synthetic_research.json --db artifacts\ui\synthetic.db
python scripts\serve_ui_local.py --db artifacts\ui\synthetic.db --host 127.0.0.1 --port 8000
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

Embedding note:

- Text embeddings now default to local-only loading.
- If `all-MiniLM-L6-v2` is not already cached locally, the pipeline falls back to deterministic hashed text embeddings instead of trying to download from Hugging Face.
- Set `SCAM_GNN_ALLOW_REMOTE_EMBEDDINGS=1` if you explicitly want remote embedding model downloads.

## UI Product Spec

The analyst and research UI product requirements are documented in [docs/scam_detection_ui_prd.md](c:\Users\user\Desktop\Ibrahims project\docs\scam_detection_ui_prd.md).

## Config-Driven Workflow

Example configs live in [examples/configs/synthetic_research.json](c:\Users\user\Desktop\Ibrahims project\examples\configs\synthetic_research.json).

An FDB-style benchmark template is also included at [examples/configs/fdb_benchmark_template.json](c:\Users\user\Desktop\Ibrahims project\examples\configs\fdb_benchmark_template.json). The benchmark track expects locally exported CSV tables; this scaffold does not download benchmark datasets automatically.

The main CLI entrypoint reads JSON configs and will:

1. load or synthesize events
2. build rolling time-based splits
3. construct heterogeneous graph tables
4. derive message and entity features
5. train baselines
6. train GNNs when `torch` and `torch-geometric` are installed
7. compute aggregate and per-channel metrics
8. generate targeted error-analysis slices

## Public Data Targets

The scaffold expects the following data sources when you move beyond synthetic data:

- SMS: UCI SMS Spam Collection
- Email: Enron and TREC Spam corpora
- URL/domain: PhiUSIIL or malicious URL corpora
- Benchmark tabular fraud: Amazon Science FDB

No public benchmark already fuses SMS, email, and URL/domain infrastructure into one graph. The custom track is intended to fill that gap.

## First Real Public Dataset

The fastest real-data path is the official UCI SMS Spam Collection.

Prepare it:

```powershell
$env:PYTHONPATH='src'
python -m scam_gnn.cli prepare-uci-sms --output data\public\uci_sms_spam_collection.csv
```

Run the research pipeline on it:

```powershell
$env:PYTHONPATH='src'
python -m scam_gnn.cli run-research --config examples\configs\uci_sms_research.json
```

Notes:

- UCI SMS Spam Collection does not include sender IDs, recipients, or native timestamps.
- This scaffold assigns deterministic synthetic timestamps in message order.
- It derives a provisional `sender_key` from phone numbers or URLs found in the message text when available, otherwise it uses `sms_sender_unknown`.
- This gives you a real public dataset connected to the existing graph pipeline, while keeping the project ready for richer email and URL datasets later.

## Second Real Dataset And Multichannel Run

The second real public dataset wired into this scaffold is the official UCI PhiUSIIL phishing URL dataset.

Prepare it:

```powershell
$env:PYTHONPATH='src'
python -m scam_gnn.cli prepare-phiusiil-url --output data\public\phiusiil_url_collection.csv
```

Run the URL-only experiment:

```powershell
$env:PYTHONPATH='src'
python -m scam_gnn.cli run-research --config examples\configs\phiusiil_url_research.json
```

Run the combined real multichannel experiment:

```powershell
$env:PYTHONPATH='src'
python -m scam_gnn.cli run-research --config examples\configs\uci_sms_phiusiil_multichannel.json
```

Build and serve the analyst UI from real multichannel data:

```powershell
python scripts\build_ui_db_local.py --config examples\configs\uci_sms_phiusiil_multichannel.json --db artifacts\ui\scam_intelligence.db --model auto
python scripts\serve_ui_local.py --db artifacts\ui\scam_intelligence.db --host 127.0.0.1 --port 8000
```

Build an HGT-backed UI database explicitly:

```powershell
python scripts\build_ui_db_local.py --config examples\configs\uci_sms_phiusiil_multichannel.json --db artifacts\ui\scam_intelligence_hgt.db --model hetero_hgt --seed 7
python scripts\serve_ui_local.py --db artifacts\ui\scam_intelligence_hgt.db --host 127.0.0.1 --port 8000
```

Notes:

- The multichannel config combines real SMS data with real phishing URL data.
- The URL dataset is capped to `15000` rows in the provided config so the baseline run stays practical in this environment.
- PhiUSIIL labels are inverted during preprocessing so `1` means malicious in this project.
- Many original numeric URL features are preserved and appended to the message feature matrix automatically.
- When public datasets do not include native timestamps, the scaffold interleaves channels and assigns synthetic timestamps so time-based splits still contain multiple channels.
- If you build the UI with `--model hetero_hgt`, the selected HGT run is written into the SQLite experiment tables even when the older summary JSON on disk does not contain HGT metrics yet.
