"""Export the toxicity model to ONNX and quantize it to int8.

Run with the *export* virtualenv (torch + transformers + optimum-onnx), not
the runtime one:

    .venv-export/bin/python tools/export_model.py

Writes into ``models/``:

    model_int8.onnx   dynamically quantized ONNX graph (the runtime artifact)
    tokenizer.json    fast tokenizer, loaded by the ``tokenizers`` package
    model_meta.json   label list and problem type, read by the service

The verification step compares the PyTorch reference logits against the float
ONNX graph and the quantized graph, using the *shipped* tokenizer with the
*shipped* truncation rule and the service's *own* decision thresholds. The
artifact is only moved into place once it passes, so a broken export cannot
be shipped by a later ``docker build``.

Why a separate virtualenv: ``optimum-onnx`` pins ``transformers<4.58`` while
the latest ``transformers`` is 5.x. The export tool is a build-time concern
and the runtime image ships no ML framework at all, so the two dependency
sets are kept apart on purpose.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import Settings  # noqa: E402
from app.model import PADDING_BUCKETS  # noqa: E402

MODEL_ID = "unitary/toxic-bert"
MAX_SEQ_LEN = 128
# Opset 18 is the minimum optimum recommends for BERT, and it is comfortably
# above the opset 14 floor imposed by scaled_dot_product_attention.
OPSET = 18

SAMPLE_TEXTS = [
    "gg, nice shot!",
    "you are trash at this game",
    "I will find you and break your legs",
    "join my discord discord.gg/abcd1234",
    "you are a disgusting idiot",
    "let's queue for ranked",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Directory to write the model artifacts into.",
    )
    parser.add_argument(
        "--opset", type=int, default=OPSET, help="ONNX opset to export with."
    )
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build"),
        help="Scratch directory for the float graph; it is not shipped.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Re-run the verification against the existing artifacts.",
    )
    return parser.parse_args()


def export_onnx(export_dir: Path, opset: int) -> Path:
    """Export the HF model to a float ONNX graph.

    ``optimum`` treats ``output`` as a directory and writes ``model.onnx``
    inside it, alongside the tokenizer it derives from the checkpoint.
    """
    from optimum.exporters.onnx import main_export

    export_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_ID} -> {export_dir} (opset {opset})")
    main_export(
        MODEL_ID,
        output=str(export_dir),
        task="text-classification",
        framework="pt",
        opset=opset,
        device="cpu",
        do_validation=False,
    )
    model_path = export_dir / "model.onnx"
    if not model_path.exists():
        raise SystemExit(f"optimum did not produce {model_path}")
    return model_path


def quantize(fp32_path: Path, int8_path: Path) -> None:
    """Apply dynamic int8 quantization to the float graph."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    print(f"Quantizing {fp32_path.name} -> {int8_path.name} (dynamic int8)")
    quantize_dynamic(
        str(fp32_path),
        str(int8_path),
        weight_type=QuantType.QInt8,
        extra_options={"MatMulConstBOnly": True},
    )


def save_tokenizer(output_dir: Path) -> None:
    """Write the fast tokenizer next to the ONNX graph."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.save_pretrained(str(output_dir))
    print(f"Saved tokenizer to {output_dir}")


def save_metadata(output_dir: Path, labels: list[str], problem_type: str) -> None:
    """Write the metadata the runtime needs to interpret the model output."""
    meta = {
        "model_id": MODEL_ID,
        "labels": labels,
        "problem_type": problem_type,
        "max_seq_len": MAX_SEQ_LEN,
        "quantization": "dynamic_int8",
        "license": "apache-2.0",
    }
    path = output_dir / "model_meta.json"
    path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Saved metadata to {path}")


def build_inputs(tokenizer_path: Path, texts: list[str]) -> dict[str, np.ndarray]:
    """Tokenize exactly the way the runtime does.

    The runtime loads ``models/tokenizer.json`` through the ``tokenizers``
    package and lets the tokenizer truncate, which keeps the trailing
    ``[SEP]``. Truncating the token ids by hand instead drops it, and that is
    a different input: it changes the scores.
    """
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    tokenizer.enable_truncation(max_length=MAX_SEQ_LEN)

    real_ids: list[list[int]] = []
    for text in texts:
        real_ids.append(list(tokenizer.encode(text).ids))

    width = max(
        next(
            (bucket for bucket in PADDING_BUCKETS if len(ids) <= bucket),
            MAX_SEQ_LEN,
        )
        for ids in real_ids
    )
    input_ids = np.zeros((len(real_ids), width), dtype=np.int64)
    attention_mask = np.zeros((len(real_ids), width), dtype=np.int64)
    token_type_ids = np.zeros((len(real_ids), width), dtype=np.int64)
    for row, ids in enumerate(real_ids):
        input_ids[row, : len(ids)] = ids
        # Only the real tokens are attended to; the padding is masked out.
        attention_mask[row, : len(ids)] = 1

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    }


def verify(
    fp32_path: Path,
    int8_path: Path,
    labels: list[str],
    tokenizer_path: Path,
) -> None:
    """Check the ONNX graphs against the PyTorch reference.

    The float graph must match PyTorch essentially exactly. The int8 graph is
    allowed to drift in the logits, but only if the drift changes no
    *decision* at the thresholds the service actually uses.

    Each sample is scored on its own, the way the API serves it. Dynamic int8
    quantization derives its activation scales from the tensor it sees, so a
    batched check would measure something the service never does.
    """
    import onnxruntime as ort
    import torch
    from transformers import AutoModelForSequenceClassification

    settings = Settings()
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID).eval()

    sessions = {
        "float": ort.InferenceSession(
            str(fp32_path), providers=["CPUExecutionProvider"]
        ),
        "int8": ort.InferenceSession(
            str(int8_path), providers=["CPUExecutionProvider"]
        ),
    }

    def sigmoid(values: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-values))

    max_logit_diff = 0.0
    max_prob_diff = 0.0
    flips: dict[str, int] = {}

    for text in SAMPLE_TEXTS:
        inputs = build_inputs(tokenizer_path, [text])
        with torch.no_grad():
            reference = model(
                input_ids=torch.tensor(inputs["input_ids"]),
                attention_mask=torch.tensor(inputs["attention_mask"]),
                token_type_ids=torch.tensor(inputs["token_type_ids"]),
            ).logits.numpy()

        logits = {}
        for name, session in sessions.items():
            feed = {
                "input_ids": inputs["input_ids"],
                "attention_mask": inputs["attention_mask"],
            }
            if "token_type_ids" in {item.name for item in session.get_inputs()}:
                feed["token_type_ids"] = inputs["token_type_ids"]
            logits[name] = session.run(None, feed)[0]

        max_logit_diff = max(
            max_logit_diff, float(np.max(np.abs(logits["float"] - reference)))
        )

        float_probs = sigmoid(logits["float"])
        int8_probs = sigmoid(logits["int8"])
        max_prob_diff = max(
            max_prob_diff, float(np.max(np.abs(float_probs - int8_probs)))
        )
        for name, threshold in (
            ("block", settings.block_threshold),
            ("review", settings.review_threshold),
        ):
            # The decision follows the strongest trait, so that is what has
            # to be compared: a label crossing the threshold on its own
            # changes nothing unless it is the strongest one.
            float_severity = np.max(float_probs, axis=-1)
            int8_severity = np.max(int8_probs, axis=-1)
            crossed = (float_severity >= threshold) != (int8_severity >= threshold)
            flips[name] = flips.get(name, 0) + int(np.count_nonzero(crossed))

    print(f"max |logit| difference, float ONNX vs PyTorch: {max_logit_diff:.6f}")
    if max_logit_diff > 1e-3:
        raise SystemExit(
            f"float ONNX export does not match PyTorch ({max_logit_diff})"
        )

    print(f"max |probability| difference, int8 vs float: {max_prob_diff:.4f}")
    if max_prob_diff > 0.10:
        raise SystemExit(f"int8 probabilities drifted too far ({max_prob_diff})")

    # The thresholds the service actually applies, not copies of them.
    for name, threshold in (
        ("block", settings.block_threshold),
        ("review", settings.review_threshold),
    ):
        print(f"  {name} threshold {threshold:.2f}: {flips.get(name, 0)} label(s) flipped")
        if flips.get(name):
            raise SystemExit(f"int8 export changes decisions at the {name} threshold")

    print("\nint8 model output on the sample set:")
    for text in SAMPLE_TEXTS:
        inputs = build_inputs(tokenizer_path, [text])
        session = sessions["int8"]
        feed = {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }
        if "token_type_ids" in {item.name for item in session.get_inputs()}:
            feed["token_type_ids"] = inputs["token_type_ids"]
        probs = sigmoid(session.run(None, feed)[0])[0]
        summary = ", ".join(
            f"{label}={value:.3f}" for label, value in zip(labels, probs, strict=True)
        )
        print(f"  {text!r:45} {summary}")


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    fp32_path = args.build_dir / "fp32" / "model.onnx"
    int8_path = output_dir / "model_int8.onnx"

    if not args.verify_only:
        fp32_path = export_onnx(args.build_dir / "fp32", args.opset)
        save_tokenizer(output_dir)

        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(MODEL_ID)
        labels = [config.id2label[index] for index in sorted(config.id2label)]
        problem_type = (
            getattr(config, "problem_type", None) or "multi_label_classification"
        )
        save_metadata(output_dir, labels, problem_type)

        # Quantize to a temporary path: the artifact is only moved into the
        # shipping directory once it has passed verification.
        pending_path = output_dir / "model_int8.onnx.pending"
        quantize(fp32_path, pending_path)
        try:
            verify(fp32_path, pending_path, labels, output_dir / "tokenizer.json")
        except BaseException:
            pending_path.unlink(missing_ok=True)
            raise
        os.replace(pending_path, int8_path)
    else:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(MODEL_ID)
        labels = [config.id2label[index] for index in sorted(config.id2label)]
        verify(fp32_path, int8_path, labels, output_dir / "tokenizer.json")

    size_mb = int8_path.stat().st_size / (1024 * 1024)
    print(f"\nDone. {int8_path} is {size_mb:.1f} MB")
    print("The float graph stays in the build directory and is not shipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
