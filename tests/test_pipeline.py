from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scam_gnn.config import load_config  # noqa: E402
from scam_gnn.pipeline import run_experiment  # noqa: E402


class PipelineTests(unittest.TestCase):
    def test_synthetic_research_pipeline_runs(self) -> None:
        config = load_config(ROOT / "examples" / "configs" / "synthetic_research.json")
        output_dir = ROOT / "artifacts" / "test_pipeline_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        config.dataset.output_dir = str(output_dir)
        summary = run_experiment(config)
        self.assertIn("logistic_regression", summary["results"])
        self.assertIn("mlp", summary["results"])
        self.assertGreater(summary["dataset"]["num_events"], 0)
        self.assertGreater(summary["dataset"]["num_entities"], 0)
        self.assertIn("error_analysis", summary)


if __name__ == "__main__":
    unittest.main()
