# Scam Detection Using Graph Neural Networks Across Multi-Channel Communications

A final-year research project. Messages from three communication channels (SMS, email and web links) are placed in
one graph together with the senders, recipients, links, websites and phone numbers they refer to. A Graph Neural
Network (GNN) then classifies every message as **scam** or **legitimate**, and is compared with two standard models
that do not use the graph.

The full explanation of the project, written for the defence, is in
[docs/PROJECT_HANDBOOK.md](docs/PROJECT_HANDBOOK.md).

## The pipeline

| Stage | File | What it does |
|---|---|---|
| 1. Data | `src/scam_gnn/data.py` | Loads the three datasets, cleans them, removes duplicates, splits 60/20/20 |
| 2. Graph construction | `src/scam_gnn/graph.py` | Creates the nodes and edges |
| 3. Features | `src/scam_gnn/features.py` | Describes every node with numbers |
| 4. Models | `src/scam_gnn/models.py` | Logistic regression, gradient-boosted trees, heterogeneous GraphSAGE |
| 5. Training | `src/scam_gnn/train.py` | Trains everything, runs the graph ablation, saves model and results |
| 6. Evaluation | `src/scam_gnn/evaluate.py` | Accuracy, precision, recall, F1, ROC-AUC, PR-AUC, confusion matrix |
| 7. Prediction | `src/scam_gnn/predict.py` | Scores one new message with the saved GNN |
| 8. Explanation | `src/scam_gnn/explain.py` | Removes one factor at a time and scores again |
| 9. Demonstration | `app/app.py` | One-page Streamlit demo |

All settings are in `config.json`.

## Data

| Channel | Dataset | Records used | Sender / recipient information |
|---|---|---|---|
| SMS | UCI SMS Spam Collection | 5,159 | none |
| Email | SpamAssassin public corpus (from the Zenodo "Phishing Email Curated Datasets" collection, CC BY 4.0) | 5,784 | yes |
| Web link | PhiUSIIL Phishing URL dataset (random sample) | 14,996 | none |

Phone calls are not included in this version; the handbook explains why.

## Results (test data, average of 3 training runs, F1-score)

| Model | SMS | Email | Web links | All channels |
|---|---|---|---|---|
| Logistic regression | 0.667 | 0.594 | 0.884 | 0.799 |
| Gradient-boosted trees | 0.911 | 0.816 | 0.968 | 0.935 |
| **Heterogeneous GraphSAGE (GNN)** | 0.870 | **0.944** | **0.982** | **0.967** |
| GNN without message-to-message edges | 0.874 | 0.921 | 0.983 | 0.963 |
| GNN with no edges at all | 0.870 | 0.809 | 0.958 | 0.922 |

The graph helps most where the data contains real relationships (email), helps a little for web links, and does not
help for SMS, where the dataset has no sender information. Every number is produced by `python -m scam_gnn.train` and
stored in `results/results.json`.

## Run it

Requirements: Python 3.11 or newer and the packages in `requirements.txt`.

```powershell
# Start the demonstration (double-clicking run_demo.bat does the same)
.\run_demo.bat

# Re-train and re-evaluate everything (about 15 minutes on a laptop)
$env:PYTHONPATH = "src"
python -W ignore -m scam_gnn.train

# Run the tests (about one minute)
python -W ignore -m unittest discover -s tests -v
```

The demo and the training need no internet connection.

## Folders

```
config.json        all settings
data/public/       the three datasets
src/scam_gnn/      the pipeline (one file per stage)
app/               the demonstration and its test inputs
results/           trained model, results.json, predictions
tests/             automated checks
docs/              the project handbook
_archive/          files from the earlier version that are no longer used (safe to delete)
```
