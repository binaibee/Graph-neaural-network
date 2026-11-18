"""Optional: download the MiniLM sentence-embedding model into ``models/``.

Only needed when ``"text_embedding": "minilm"`` is set in config.json.  Requires
an internet connection once and the ``sentence-transformers`` package:

    python -m pip install sentence-transformers
    python -m scam_gnn.download_model
"""
from __future__ import annotations

from scam_gnn.config import load_config


def main() -> None:
    from sentence_transformers import SentenceTransformer

    config = load_config()
    target = config.path(config.minilm_path)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    target.mkdir(parents=True, exist_ok=True)
    model.save(str(target))
    print(f"Saved to {target}")


if __name__ == "__main__":
    main()
