"""ONNX Runtime toxicity scorer.

The model is a BERT-base sequence classifier (``unitary/toxic-bert``) exported
to ONNX and dynamically quantized to int8. It runs on CPU only, with a fixed
thread count so that several uvicorn workers do not fight over the cores.

Two details matter for latency:

* **Padding buckets.** ONNX Runtime re-optimizes a graph the first time it
  sees a new input shape. Chat messages vary in length, so padding every
  message to a fixed 128 tokens would waste compute, while padding to the
  exact length would trigger a re-optimization per length. Padding up to one
  of a few fixed buckets bounds both cost and re-optimizations.
* **Score cache.** "gg", "noob", "lol" repeat constantly; caching their
  probabilities removes the model from the hot path entirely.
"""

from __future__ import annotations

import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

# Padding buckets, in tokens. The first one that fits a message is used.
PADDING_BUCKETS = (16, 32, 64, 128)


class ModelLoadError(RuntimeError):
    """Raised when the model artifacts are missing or unusable."""


class ToxicityModel:
    """Scores text with the quantized ONNX toxicity classifier."""

    def __init__(
        self,
        model_dir: Path,
        onnx_file: str,
        tokenizer_file: str,
        labels_file: str,
        threads: int = 2,
        max_seq_len: int = 128,
        cache_size: int = 2048,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._onnx_file = onnx_file
        self._tokenizer_file = tokenizer_file
        self._labels_file = labels_file
        self._threads = threads
        self._max_seq_len = max_seq_len
        self._cache_size = cache_size

        self._session: ort.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._labels: list[str] = []
        self._problem_type = "multi_label_classification"
        self._model_id: str | None = None
        self._input_names: set[str] = set()
        self._output_name = ""

        self._cache: OrderedDict[str, dict[str, float]] = OrderedDict()
        self._lock = threading.Lock()

    # --- properties --------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._session is not None

    @property
    def model_id(self) -> str | None:
        return self._model_id

    @property
    def labels(self) -> list[str]:
        return list(self._labels)

    # --- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        """Load the ONNX graph, the tokenizer and the metadata."""
        onnx_path = Path(self._onnx_file)
        if not onnx_path.is_absolute():
            onnx_path = self._model_dir / self._onnx_file
        tokenizer_path = self._model_dir / self._tokenizer_file
        meta_path = self._model_dir / self._labels_file
        for path in (onnx_path, tokenizer_path, meta_path):
            if not path.exists():
                raise ModelLoadError(f"missing model artifact: {path}")

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._labels = list(meta["labels"])
        self._problem_type = str(
            meta.get("problem_type", "multi_label_classification")
        )
        self._model_id = str(meta.get("model_id", ""))
        self._max_seq_len = min(
            self._max_seq_len, int(meta.get("max_seq_len", self._max_seq_len))
        )

        options = ort.SessionOptions()
        options.intra_op_num_threads = self._threads
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self._session = ort.InferenceSession(
            str(onnx_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        # Truncation has to be enabled on the tokenizer, not applied to the
        # token ids afterwards: slicing the ids drops the trailing [SEP],
        # which is a different input from the one the export verified and it
        # changes the scores.
        self._tokenizer.enable_truncation(max_length=self._max_seq_len)
        self._input_names = {item.name for item in self._session.get_inputs()}
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            "loaded model %s (%d labels, threads=%d, max_seq_len=%d)",
            self._model_id,
            len(self._labels),
            self._threads,
            self._max_seq_len,
        )

    def warmup(self) -> None:
        """Run one dummy inference so the first real message is not the slowest."""
        if self._session is None:
            return
        self.score("warmup")
        logger.info("model warmup complete")

    # --- scoring -----------------------------------------------------------

    def score(self, text: str) -> dict[str, float]:
        """Label probabilities for a single message, using the cache."""
        cached = self._cache_get(text)
        if cached is not None:
            return cached
        scores = self.score_batch([text])[0]
        self._cache_put(text, scores)
        return scores

    def score_batch(self, texts: list[str]) -> list[dict[str, float]]:
        """Label probabilities for a batch of messages."""
        if not texts:
            return []
        if self._session is None or self._tokenizer is None:
            raise ModelLoadError("model is not loaded")

        results: list[dict[str, float]] = [{} for _ in texts]
        groups: dict[int, list[tuple[int, list[int]]]] = {}
        for index, text in enumerate(texts):
            encoding = self._tokenizer.encode(text)
            ids = list(encoding.ids)
            groups.setdefault(self._bucket_for(len(ids)), []).append((index, ids))

        for bucket, group in groups.items():
            input_ids = np.zeros((len(group), bucket), dtype=np.int64)
            attention_mask = np.zeros((len(group), bucket), dtype=np.int64)
            for row, (_, ids) in enumerate(group):
                input_ids[row, : len(ids)] = ids
                attention_mask[row, : len(ids)] = 1

            feed = {"input_ids": input_ids, "attention_mask": attention_mask}
            if "token_type_ids" in self._input_names:
                feed["token_type_ids"] = np.zeros((len(group), bucket), dtype=np.int64)

            logits = self._session.run([self._output_name], feed)[0]
            probabilities = self._to_probabilities(logits)
            for row, (index, _) in enumerate(group):
                results[index] = {
                    label: float(value)
                    for label, value in zip(
                        self._labels, probabilities[row], strict=True
                    )
                }
        return results

    # --- internals ---------------------------------------------------------

    def _bucket_for(self, length: int) -> int:
        for bucket in PADDING_BUCKETS:
            if length <= bucket:
                return bucket
        return self._max_seq_len

    def _to_probabilities(self, logits: np.ndarray) -> np.ndarray:
        """Turn raw logits into probabilities for the model's problem type."""
        if self._problem_type == "multi_label_classification":
            return 1.0 / (1.0 + np.exp(-logits))
        shifted = logits - logits.max(axis=-1, keepdims=True)
        exponentials = np.exp(shifted)
        return exponentials / exponentials.sum(axis=-1, keepdims=True)

    def _cache_get(self, key: str) -> dict[str, float] | None:
        with self._lock:
            value = self._cache.get(key)
            if value is not None:
                self._cache.move_to_end(key)
            return value

    def _cache_put(self, key: str, value: dict[str, float]) -> None:
        with self._lock:
            self._cache[key] = value
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)


def build_model(settings) -> ToxicityModel:
    """Construct a :class:`ToxicityModel` from the service settings."""
    return ToxicityModel(
        model_dir=settings.model_dir,
        onnx_file=settings.onnx_file,
        tokenizer_file=settings.tokenizer_file,
        labels_file=settings.labels_file,
        threads=settings.onnx_threads,
        max_seq_len=settings.max_seq_len,
        cache_size=settings.score_cache_size,
    )
