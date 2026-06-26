"""
Offline text embedding database for NCH.

Reads precomputed pangu text encoder embeddings from a SQLite cache.
The cache is built by a separate offline script (see scripts/ directory).
At training time, EmbeddingDB.get_embedding(text, model_id) returns
{"encoder_hidden_states": tensor of shape [B, 256, 1536]}.

NOTE: This module intentionally avoids importing torch_npu or the pangu
inference stack, so it can be imported in any environment.
"""
from __future__ import annotations

import io
import sqlite3
from types import SimpleNamespace
from typing import Any, Mapping, Optional

import torch

from framework.instantiate import instantiate


DDL = """
CREATE TABLE IF NOT EXISTS embeddings (
    text TEXT NOT NULL,
    model_id TEXT NOT NULL,
    data BLOB NOT NULL,
    PRIMARY KEY (text, model_id)
);
"""


class EmbeddingDB:
    """
    Offline embedding cache backed by SQLite.

    Parameters
    ----------
    trainable_vector_cfg : optional
        Config for TrainableVector_multitask, used to prepend task token
        prefixes (e.g. "p_dehalo_0,...,p_dehalo_9. ") to the prompt before
        lookup. During training the vlm_text_encoder is a dummy
        SimpleNamespace(tokenizer=None, model=None) since only the
        task_tokens dict is needed.
    task_tokens : str, optional
        Task key (e.g. "dehalo") selecting which prefix to prepend.
    embedding_cache_path : str
        Path to the SQLite cache file.
    """

    def __init__(
        self,
        trainable_vector_cfg: Optional[Mapping[str, Any]] = None,
        task_tokens: Optional[str] = None,
        embedding_cache_path: str = "./embedding_cache_pangu_1b_vl.sqlite",
    ):
        if trainable_vector_cfg:
            trainable_vector_cfg["params"]["vlm_text_encoder"] = SimpleNamespace(
                tokenizer=None, model=None
            )
            self.trainable_vector = instantiate(trainable_vector_cfg)
        else:
            self.trainable_vector = None
        self.task_tokens = task_tokens
        self.cache_path = embedding_cache_path
        print(f"Embedding offline cache at {self.cache_path}")
        self.conn = sqlite3.connect(self.cache_path)
        self.conn.execute(DDL)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("PRAGMA busy_timeout=5000;")

    def get_embedding(self, text, model_id, device="cpu"):
        if isinstance(text, str):
            text = [text]
        if self.trainable_vector:
            text = [
                self.trainable_vector.task_tokens[self.task_tokens] + cap
                if self.task_tokens
                else cap
                for cap in text
            ]

        results = []
        for t in text:
            cur = self.conn.execute(
                "SELECT data FROM embeddings WHERE text=? AND model_id=?",
                (t, model_id),
            )
            row = cur.fetchone()
            if row:
                tensor = torch.load(io.BytesIO(row[0]), map_location="cpu")
                results.append(tensor.unsqueeze(0))
            else:
                print(f"WARNING: No offline embedding! text: {t}, model: {model_id}")
                if results:
                    zero_tensor = torch.zeros_like(results[0])
                else:
                    zero_tensor = torch.zeros(1, 256, 1536)
                results.append(zero_tensor)

        results = torch.cat(results, dim=0).to(device) if results else None
        return {"encoder_hidden_states": results}
