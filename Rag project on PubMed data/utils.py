import json
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from pathlib import Path
import yaml



_cfg = yaml.safe_load((Path(__file__).parent / "config.yml").read_text())

ST_CONFIGS = _cfg["ST_CONFIGS"]
MEDCPT = _cfg["MEDCPT"]



def load_corpus(path):
    with open(path, encoding="utf-8") as f:
        text = f.read().strip()
    records = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(l) for l in text.splitlines() if l.strip()]
    )
    for r in records:
        r["pmid"] = str(r.get("pmid") or r.get("pubmed_id"))
    return {r["pmid"]: r for r in records if r.get("abstract")}


class QueryEncoder:
    """Encodes queries the way each model expects. Returns normalized vectors."""

    def __init__(self, key):
        self.key = key
        if key == "medcpt":
            self.tok = AutoTokenizer.from_pretrained(MEDCPT["query_encoder"])
            self.model = (
                AutoModel.from_pretrained(MEDCPT["query_encoder"]).to(DEVICE).eval()
            )
        else:
            self.cfg = ST_CONFIGS[key]
            self.model = SentenceTransformer(self.cfg["name"], device=DEVICE)

    def encode(self, queries):
        if self.key == "medcpt":
            with torch.no_grad():
                enc = self.tok(
                    queries,
                    truncation=True,
                    padding=True,
                    max_length=MEDCPT["query_max_length"],
                    return_tensors="pt",
                ).to(DEVICE)
                cls = self.model(**enc).last_hidden_state[:, 0, :]
                return torch.nn.functional.normalize(cls, dim=-1).cpu().numpy()
        if "query_prompt_name" in self.cfg:
            return self.model.encode(
                queries,
                prompt_name=self.cfg["query_prompt_name"],
                normalize_embeddings=True,
            )
        prefix = self.cfg.get("query_prefix", "")
        return self.model.encode(
            [prefix + q for q in queries], normalize_embeddings=True
        )
