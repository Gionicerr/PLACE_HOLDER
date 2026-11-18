"""Run participant 02's A123 experiment with the team's unchanged train.py."""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from generate import prepare_sft_files, quality
from protocol import (CONFIG, DATA, ROOT, atomic_text, chat_prompt, read_json, read_jsonl,
                      read_split, save_json, save_jsonl, score, user_prompt)

PLAN = read_json(ROOT / "configs" / "A123.json")


def train_command(train_file, val_file, work, device):
    return [sys.executable, str(ROOT / "train.py"),
            "--train-file", str(train_file), "--validation-file", str(val_file),
            "--output-dir", str(work), "--model", PLAN["student"], "--revision", PLAN["revision"],
            "--seed", str(PLAN["seed"]), "--epochs", str(PLAN["epochs"]),
            "--batch-size", str(PLAN["micro_batch_size"]), "--gradient-accumulation-steps",
            str(PLAN["effective_batch_size"] // PLAN["micro_batch_size"]),
            "--learning-rate", str(PLAN["learning_rate"]), "--max-length", str(PLAN["max_length"]),
            "--lora-rank", str(PLAN["rank"]), "--lora-alpha", str(PLAN["alpha"]),
            *(["--cpu"] if device == "cpu" else [])]


def train_once(train_file, val_file, dest, device, threads):
    target = dest / "training"
    if target.is_dir():
        return
    with TemporaryDirectory(prefix="sft-", dir=dest) as temp:
        work = Path(temp) / "result"
        command = train_command(train_file, val_file, work, device)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}
        started = time.perf_counter()
        attempt = {"started_utc": datetime.now(timezone.utc).isoformat(), "status": "interrupted"}
        try:
            subprocess.run(command, env=env, check=True)
            for name in ("adapter/adapter_model.safetensors", "adapter/adapter_config.json",
                         "trainer_state.json", "run_results.json", "run_config.json"):
                if not (work / name).is_file():
                    raise FileNotFoundError(f"Shared trainer did not produce {name}")
            work.rename(target)
            attempt["status"] = "completed"
        except BaseException as error:
            attempt["error"] = type(error).__name__
            raise
        finally:
            attempt["elapsed_seconds"] = time.perf_counter() - started
            with (dest / "training_attempts.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(attempt) + "\n")


def predictions(model, tokenizer, test, tag, dest):
    import torch
    from transformers import GenerationConfig

    path = dest / f"{tag}.predictions.jsonl"
    if path.exists():
        cached = read_jsonl(path)
        if len(cached) != len(test):
            raise ValueError(f"Incomplete {path}")
        for rec, source in zip(cached, test):
            if (rec["prompt_id"], rec["reference_answer"]) != (source["id"], source["reference_answer"]):
                raise ValueError(f"Unexpected test example in {path}")
            if score(rec["response"], source["reference_answer"]) != (rec["extracted_answer"], rec["is_correct"]):
                raise ValueError(f"Invalid cached test score in {path}")
        return cached

    tokenizer.padding_side = "left"
    model.eval()
    config = GenerationConfig(do_sample=False, max_new_tokens=CONFIG["generation"]["max_new_tokens"],
                              pad_token_id=tokenizer.pad_token_id, eos_token_id=model.generation_config.eos_token_id)
    rows = []
    batch_size = PLAN["eval_batch_size"]
    with torch.inference_mode(), (model.disable_adapter() if tag == "baseline" else nullcontext()):
        for start in range(0, len(test), batch_size):
            batch = test[start:start + batch_size]
            inputs = tokenizer([chat_prompt(r["question"]) for r in batch], padding=True,
                               add_special_tokens=False, return_tensors="pt").to(model.device)
            output = model.generate(**inputs, generation_config=config)
            generated = output[:, inputs.input_ids.shape[1]:]
            for source, response in zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                response = response.strip()
                extracted, correct = score(response, source["reference_answer"])
                rows.append({"prompt_id": source["id"], "prompt": user_prompt(source["question"]),
                             "reference_answer": source["reference_answer"], "response": response,
                             "extracted_answer": extracted, "is_correct": correct})
            print(f"{tag} | {len(rows)}/{len(test)}", flush=True)
    save_jsonl(path, rows)
    return rows


def finish_report(dest, stage, alignment, n_val, teacher_metrics, test_metrics, device):
    training = read_json(dest / "training" / "run_results.json")
    baseline, student = test_metrics["baseline"], test_metrics["student"]
    delta = 100 * (student["accuracy"] - baseline["accuracy"])
    metrics = {"stage": stage, "seed": PLAN["seed"], "teacher_A": teacher_metrics,
               "train_samples": alignment["n_sft"], "validation_samples": n_val,
               **test_metrics, "delta_pp": delta, "sft": training}
    save_json(dest / "metrics.json", metrics)
    notes = (
        f"# Эксперимент A123 · {stage}\n\n"
        f"Teacher A ({CONFIG['teachers']['A']['model']}, Q4_K_M) решил правильно "
        f"{teacher_metrics['correct']} из {teacher_metrics['n']} задач. "
        f"Парсер отклонил {teacher_metrics['invalid']} ответов. В одинаковую для трёх Teacher "
        f"обучающую выборку вошло {alignment['n_sft']} задач. Для каждой сохранены расчёт и финальный ответ модели.\n\n"
        f"Исходная Student {PLAN['student']} прошла {PLAN['epochs']} эпохи LoRA SFT с seed {PLAN['seed']}, "
        f"rank {PLAN['rank']}, alpha {PLAN['alpha']} и эффективным батчем {PLAN['effective_batch_size']}. "
        f"Общий train.py маскировал входную часть диалога и padding. Средний train loss — "
        f"{training['train_loss']:.4f}. Для {n_val} валидационных задач целевым текстом была строка "
        f"с эталонным числом, итоговый validation loss — {training['eval_loss']:.4f}. "
        f"Обучение заняло {training['train_time_seconds']/60:.1f} мин на {device}.\n\n"
        f"Из {student['n']} тестовых задач исходная модель решила {baseline['correct']} "
        f"({baseline['accuracy']:.1%}), после обучения — {student['correct']} ({student['accuracy']:.1%}). "
        f"Изменение — {delta:+.1f} п. п. Нарушений формата соответственно {baseline['invalid']} "
        f"и {student['invalid']}. Ошибки формата учитывались как неверные ответы. "
        f"Для вывода о влиянии Teacher потребуются второй seed и результаты остальных Teacher.\n")
    atomic_text(dest / "notes.md", notes)
    print(notes)


def run(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    folder = args.output / args.stage
    dest = folder / "A123"
    sources = read_split(args.data, "train", args.stage)
    alignment = prepare_sft_files(folder, sources)
    train_file = folder / "sft" / "teacher_A.jsonl"
    validation = read_split(args.data, "validation", args.stage)
    val_file = dest / "validation.jsonl"
    save_jsonl(val_file, [{"prompt_id": r["id"], "prompt": user_prompt(r["question"]),
                          "response": "#### " + r["reference_answer"]} for r in validation])
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; use --device cpu")
    torch.set_num_threads(args.threads)
    set_seed(PLAN["seed"])
    manifest = {"stage": args.stage, "sft": PLAN, "device": device, "threads": args.threads,
                "processor": platform.processor(), "platform": platform.platform(), "python": sys.version.split()[0],
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
                "teacher_sft_sha256": hashlib.sha256(train_file.read_bytes()).hexdigest(),
                "source_splits": CONFIG["data"], "versions": {p: version(p) for p in
                ("torch", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors")}}
    manifest_path = dest / "experiment.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Inputs, software or device changed. Use a separate output directory.")
    save_json(manifest_path, manifest)
    if (dest / "notes.md").is_file() and (dest / "metrics.json").is_file():
        print((dest / "notes.md").read_text(encoding="utf-8"))
        return

    tokenizer = AutoTokenizer.from_pretrained(PLAN["student"], revision=PLAN["revision"], use_fast=True)
    native = tokenizer.apply_chat_template([{"role": "user", "content": user_prompt("format check")}],
                                           tokenize=False, add_generation_prompt=True)
    if native != chat_prompt("format check"):
        raise ValueError("The fixed generation prompt differs from the shared trainer's native template")
    train_once(train_file, val_file, dest, device, args.threads)
    base = AutoModelForCausalLM.from_pretrained(PLAN["student"], revision=PLAN["revision"],
                                                use_safetensors=True).to(device)
    model = PeftModel.from_pretrained(base, dest / "training" / "adapter")
    test = read_split(args.data, "test", args.stage)
    test_metrics = {}
    for name in ("baseline", "student"):
        rows = predictions(model, tokenizer, test, name, dest)
        correct = sum(r["is_correct"] for r in rows)
        invalid = sum(r["extracted_answer"] is None for r in rows)
        test_metrics[name] = {"n": len(rows), "correct": correct, "accuracy": correct / len(rows),
                              "invalid": invalid, "invalid_format_rate": invalid / len(rows)}
    finish_report(dest, args.stage, alignment, len(validation),
                  quality(read_jsonl(folder / "teacher_A.jsonl")), test_metrics, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    from filelock import FileLock

    dest = args.output / args.stage / "A123"
    dest.mkdir(parents=True, exist_ok=True)
    with FileLock(str(dest / "run.lock"), timeout=0):
        run(args)


if __name__ == "__main__":
    main()
