# Live demonstration

A Streamlit interface on top of the existing research code in `src/scam_gnn`. It does not replace or change that code: it
trains the project's heterogeneous GraphSAGE with the existing training loop, saves the weights, and uses them to score
messages you type.

## Start the demo

Double-click **`run_demo.bat`** in the project folder (or run it from a terminal). The window first prints
"Preparing the data and checking the model" (about 10 seconds; about 40 seconds the very first time or after the data
or code changes), then your browser opens at <http://localhost:8501> and the page is ready about 15 seconds later.

Start it a few minutes before you present, and keep the laptop plugged in: on battery saver everything runs slower.

Equivalent command, from the project folder:

```powershell
python -m streamlit run demo\app.py --server.headless true --server.address localhost --browser.gatherUsageStats false --theme.base light
```

No internet connection is needed. Everything runs from the files in this folder.

## What to click during the presentation

1. **1 · Try a message** — click the example button **Scam with a phone number** (or type a message and press **Analyze Message**).
   Read out the coloured verdict, the risk score, the "Why?" lines and the graph.
2. Click the example button **Scam with no number or link**. The model misses it; use it to explain what the model relies on.
3. **2 · Run the tests** (Test B) — real test-set records with their true labels. Press **Pick 12 different records** to draw a
   different random sample.
4. **2 · Run the tests** (Test A) — press **Run the 10 tests** (about 25 seconds). Expected result: 7 of 10 correct.
5. **3 · Results** — metrics, confusion matrix and the comparison of all models.
6. **3 · Results**, *Limitations* section — graph sizes and the limitations list.

If anything looks wrong, press F5. There is also a **Reload model and data** button in the sidebar (click the » arrow at the top left to open it).

## Files

| File | Purpose |
|---|---|
| `app.py` | The Streamlit interface |
| `engine.py` | Adds a typed message to the graph and scores it with the saved GNN |
| `graph_view.py` | Draws the graph around a message |
| `train_demo_model.py` | Trains the GNN with the project's training loop and saves it to `model/` |
| `model/hetero_graphsage.pt` | The trained GNN weights |
| `model/meta.json` | Training details, feature list, test metrics |
| `model/predictions.csv` | The GNN's score for every message, with split and label |
| `model/prepared_graph.pkl` | The graph and features built by the research code, saved so the demo starts quickly. Rebuilt automatically if the data, config or code change; safe to delete. |
| `model/experiment_summary.json` | Copy of the output of the project's `run-research` command (all models, 3 seeds) |
| `sample_data/test_cases.json` | The ten hand-written test messages |
| `tests/test_engine.py` | Checks that the demo gives the same scores as the original pipeline |

## Retrain or regenerate

```powershell
# Retrain the demo model (about 2 minutes). Results are deterministic for a fixed seed.
python -W ignore demo\train_demo_model.py

# Re-run the full experiment for all models and seeds (about 6 minutes), then refresh the demo's copy.
$env:PYTHONPATH='src'
python -W ignore -m scam_gnn.cli run-research --config examples\configs\uci_sms_phiusiil_multichannel.json
Copy-Item artifacts\uci_sms_phiusiil_multichannel\uci_sms_phiusiil_multichannel_summary.json demo\model\experiment_summary.json

# Check the demo against the research code (several minutes).
python -W ignore -m unittest discover -s demo\tests -v
```

## What the demo does and does not do

- **Typed input is SMS only.** A typed message becomes a new `message` node, linked to sender, recipient, phone, URL and
  domain nodes by the project's own `build_graph_tables`. The trained GNN then runs over the graph and the score of the
  new node is shown.
- **URLs are shown from the test set, not typed.** The URL inputs include about 50 page statistics from the PhiUSIIL
  dataset (lines of code, number of images, similarity index and so on) that cannot be computed for a new link offline.
- **A typed message is not saved.** Each analysis starts from the same stored graph.
- **Speed.** About 2 to 3 seconds per message on this laptop. To stay fast, the demo updates only the part of the graph
  the new message touches; `tests/test_engine.py` confirms this gives the same score as rebuilding the whole graph with
  the research code.
- **Confusion matrix, accuracy and F1** are not in the original code. The demo derives them from the same test
  predictions and says so on screen.
- **"Relevant risk indicators"** are the model's inputs and facts from the graph. No explainability method is implemented.

## Requirements

Python 3.11+, `numpy`, `pandas`, `torch`, `torch-geometric`, `scikit-learn`, `streamlit`, `matplotlib`. All are already
installed on the machine this was built on (Python 3.14, PyTorch 2.12, PyTorch Geometric 2.8, Streamlit 1.55).

## Troubleshooting

| Problem | Fix |
|---|---|
| "Missing demo\model\..." | Run `python demo\train_demo_model.py` |
| "embedding backend changed" or "feature columns differ" | A package was installed or removed since training. Retrain. |
| Port already in use | Close the other terminal window running the demo, or add `--server.port 8502` |
| Page stays on "Loading…" for over a minute, or goes grey | Refresh the browser tab (F5). If it persists, close the terminal window and start `run_demo.bat` again. |
| Browser does not open | Type `http://localhost:8501` into the browser yourself |
