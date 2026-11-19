"""Evaluate the team's original Student and optional LoRA adapters on the fixed split."""

import argparse
import csv
import hashlib
import io
import re
import statistics
import time
from contextlib import nullcontext
from importlib.metadata import version
from pathlib import Path

from protocol import (CONFIG, DATA, ROOT, atomic_text, chat_prompt, read_json,
                      read_split, save_json, save_jsonl, score, user_prompt)

PLAN = read_json(ROOT / "configs" / "A123.json")
MODEL, REVISION = PLAN["student"], PLAN["revision"]
MAX_NEW_TOKENS = CONFIG["generation"]["max_new_tokens"]


def adapter_arg(text):
    label, _, path = text.partition("=")
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", label) or label.lower() == "baseline" or not path:
        raise argparse.ArgumentTypeError("expected LABEL=adapter_directory (baseline is reserved)")
    return label, Path(path)


def adapter_metadata(path):
    config = read_json(path / "adapter_config.json")
    weights = path / "adapter_model.safetensors"
    if (config.get("peft_type"), config.get("base_model_name_or_path"), config.get("revision")) != (
        "LORA", MODEL, REVISION
    ):
        raise ValueError(f"{path} must be a LoRA adapter for {MODEL} at {REVISION}")
    with weights.open("rb") as stream:
        metadata = {"adapter_sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    run_config, run_results = path.parent / "run_config.json", path.parent / "run_results.json"
    if run_config.is_file():
        metadata["seed"] = read_json(run_config).get("seed")
    if run_results.is_file():
        metadata["train_time_seconds"] = read_json(run_results).get("train_time_seconds")
    return metadata


def evaluate_model(model, tokenizer, questions, label, batch_size, context):
    import torch
    from transformers import GenerationConfig

    eos_token_id = model.generation_config.eos_token_id
    eos_ids = frozenset(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
    config = GenerationConfig(
        do_sample=False, num_beams=1, max_new_tokens=MAX_NEW_TOKENS,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=eos_token_id,
    )
    predictions = []
    started = time.perf_counter()
    with torch.inference_mode(), context:
        for start in range(0, len(questions), batch_size):
            batch = questions[start:start + batch_size]
            inputs = tokenizer([chat_prompt(row["question"]) for row in batch], padding=True,
                               add_special_tokens=False, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, generation_config=config)
            generated = output[:, inputs.input_ids.shape[1]:].tolist()
            for row, token_ids in zip(batch, generated, strict=True):
                length = next((i for i, token in enumerate(token_ids) if token in eos_ids), len(token_ids))
                response = tokenizer.decode(token_ids[:length], skip_special_tokens=True).strip()
                extracted, correct = score(response, row["reference_answer"])
                predictions.append({
                    "prompt_id": row["id"], "reference_answer": row["reference_answer"],
                    "response": response, "extracted_answer": extracted, "is_correct": correct,
                    "response_tokens": length, "hit_token_limit": length == MAX_NEW_TOKENS,
                })
            if len(predictions) == len(questions) or (start // batch_size + 1) % 5 == 0:
                print(f"{label}: {len(predictions)}/{len(questions)}", flush=True)
    return predictions, time.perf_counter() - started


def summarize(rows):
    n = len(rows)
    correct = sum(row["is_correct"] for row in rows)
    invalid = sum(row["extracted_answer"] is None for row in rows)
    lengths = [row["response_tokens"] for row in rows]
    return {
        "n": n, "correct": correct, "accuracy": correct / n,
        "wrong_number": n - correct - invalid, "invalid": invalid, "invalid_format_rate": invalid / n,
        "token_limit_count": sum(row["hit_token_limit"] for row in rows),
        "response_tokens_mean": statistics.mean(lengths),
        "response_tokens_median": statistics.median(lengths), "response_tokens_max": max(lengths),
    }


def run(data, output, adapters=(), split="test", batch_size=8, device="auto"):
    import torch
    from filelock import FileLock
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if split not in ("test", "validation"):
        raise ValueError("split must be test or validation")
    adapters = list(adapters)
    if len({label.lower() for label, _ in adapters}) != len(adapters):
        raise ValueError("Adapter labels must be distinct")
    adapter_info = {label: adapter_metadata(path) for label, path in adapters}
    stage = "pilot" if split == "validation" else "full"
    if data.name == f"{split}.jsonl" and data.is_file():
        data = data.parent
    questions = read_split(data, split, stage)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")
    if device == "cpu":
        torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")

    output.mkdir(parents=True, exist_ok=True)
    with FileLock(str(output / ".evaluate.lock"), timeout=0):
        if any(path.name != ".evaluate.lock" for path in output.iterdir()):
            raise ValueError(f"Use a new or empty output directory: {output}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side="left")
        native = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt("Template check")}],
                                               tokenize=False, add_generation_prompt=True)
        if native != chat_prompt("Template check"):
            raise ValueError("The tokenizer chat template differs from the team's protocol")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL, revision=REVISION, use_safetensors=True, torch_dtype=torch.float32,
        ).to(device).eval()
        if adapters:
            from peft import PeftModel

            first_label, first_adapter = adapters[0]
            model = PeftModel.from_pretrained(model, first_adapter, adapter_name=first_label).eval()
            for label, path in adapters[1:]:
                model.load_adapter(path, adapter_name=label, is_trainable=False)

        metrics = {}
        for label in ["baseline", *adapter_info]:
            context = nullcontext()
            if adapters:
                if label == "baseline":
                    context = model.disable_adapter()
                else:
                    model.set_adapter(label)
            rows, elapsed = evaluate_model(model, tokenizer, questions, label, batch_size, context)
            save_jsonl(output / f"{label}.predictions.jsonl", rows)
            metrics[label] = {**summarize(rows), **adapter_info.get(label, {}),
                              "inference_seconds": round(elapsed, 3)}
            metrics[label]["delta_pp"] = round(100 * (metrics[label]["accuracy"] - metrics["baseline"]["accuracy"]), 8)
            result = metrics[label]
            print(f"{label}: {result['correct']}/{result['n']} = {result['accuracy']:.1%}, "
                  f"invalid: {result['invalid']}, delta: {result['delta_pp']:+.1f} pp", flush=True)

        manifest = {
            "model": MODEL, "revision": REVISION, "split": split, "stage": stage,
            "source_split_sha256": CONFIG["data"][split]["sha256"],
            "prompt_system": CONFIG["generation"]["system"], "prompt_instruction": CONFIG["generation"]["instruction"],
            "decoding": "greedy", "max_new_tokens": MAX_NEW_TOKENS, "batch_size": batch_size,
            "dtype": "float32", "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
            "versions": {name: version(name) for name in ("torch", "transformers", "peft", "accelerate")},
        }
        fields = ["model", "split", "stage", "seed", "n", "correct", "accuracy", "delta_pp",
                  "wrong_number", "invalid", "invalid_format_rate", "token_limit_count",
                  "response_tokens_mean", "response_tokens_median", "response_tokens_max",
                  "train_time_seconds", "inference_seconds"]
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for label, result in metrics.items():
            writer.writerow({"model": label, "split": split, "stage": stage, **result})
        atomic_text(output / "metrics.csv", stream.getvalue())
        report = {"protocol": manifest, "models": metrics}
        save_json(output / "metrics.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA,
                        help="original project_data.zip, original test.jsonl, or split directory")
    parser.add_argument("--output", type=Path, required=True, help="new or empty results directory")
    parser.add_argument("--adapter", type=adapter_arg, action="append", default=[], metavar="LABEL=DIR")
    parser.add_argument("--split", choices=("test", "validation"), default="test",
                        help="test: all 200 held-out tasks; validation: 30 smoke-test tasks")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()
    run(args.data, args.output, args.adapter, args.split, args.batch_size, args.device)


if __name__ == "__main__":
    main()
