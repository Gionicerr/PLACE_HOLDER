"""Member 02: LoRA SFT on Teacher A, followed by paired baseline and Student evaluation."""

import argparse
import hashlib
import json
import platform
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory

from generate import quality, settings, validate_record
from protocol import (CONFIG, DATA, ROOT, atomic_text, chat_prompt, read_json, read_jsonl,
                      read_split, save_json, save_jsonl, score)

PLAN = read_json(ROOT / "configs" / "A123.json")


def teacher_set(path, sources):
    rows = read_jsonl(path)
    if len(rows) != len(sources):
        raise ValueError(f"{path} contains {len(rows)} responses for {len(sources)} prompts")
    first = rows[0]["generation_config"]
    cfg = settings(PLAN["teacher"], first["threads"], first["gpu_layers"])
    for row, source in zip(rows, sources):
        validate_record(row, source, PLAN["teacher"], cfg)
    return rows


def encode_example(tokenizer, question, answer, limit):
    prompt = tokenizer.encode(chat_prompt(question), add_special_tokens=False)
    target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
    room = limit - len(prompt)
    if room < min(2, len(target)):
        raise ValueError("The complete question and an assistant target do not fit max_length")
    truncated = len(target) > room
    target = target[-room:]
    return {"input_ids": prompt + target, "labels": [-100] * len(prompt) + target}, truncated


def padded_batch(features, pad_token_id):
    import torch

    width = max(len(f["input_ids"]) for f in features)
    ids, labels, masks = [], [], []
    for feature in features:
        n = len(feature["input_ids"])
        padding = width - n
        ids.append(feature["input_ids"] + [pad_token_id] * padding)
        labels.append(feature["labels"] + [-100] * padding)
        masks.append([1] * n + [0] * padding)
    return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(masks)}


def last_complete_checkpoint(folder):
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "adapter_model.safetensors", "adapter_config.json")
    valid = [p for p in folder.glob("checkpoint-*") if p.name[11:].isdigit()
             and all((p / name).is_file() for name in required)]
    return str(max(valid, key=lambda p: int(p.name[11:]))) if valid else None


def train(base, tokenizer, train_rows, validation, dest, device):
    import torch
    from functools import partial
    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments

    train_data, val_data = [], []
    truncated = {"train": 0, "validation": 0}
    for split, data, output in (("train", train_rows, train_data),
                                ("validation", validation, val_data)):
        for row in data:
            if split == "train":
                question, answer = row["prompt"], row["response"]
            else:
                question, answer = row["question"], "#### " + row["reference_answer"]
            features, cut = encode_example(tokenizer, question, answer, PLAN["max_length"])
            output.append(features)
            truncated[split] += int(cut)

    base.config.use_cache = False
    model = get_peft_model(base, LoraConfig(
        r=PLAN["rank"], lora_alpha=PLAN["alpha"], lora_dropout=PLAN["dropout"],
        target_modules=PLAN["target_modules"], bias="none", task_type="CAUSAL_LM",
    ))
    checkpoints = dest / "checkpoints"
    args = TrainingArguments(
        output_dir=str(checkpoints), use_cpu=device == "cpu", report_to="none", seed=PLAN["seed"],
        data_seed=PLAN["seed"], num_train_epochs=PLAN["epochs"],
        per_device_train_batch_size=PLAN["micro_batch_size"],
        gradient_accumulation_steps=PLAN["effective_batch_size"] // PLAN["micro_batch_size"],
        per_device_eval_batch_size=PLAN["micro_batch_size"], learning_rate=PLAN["learning_rate"],
        lr_scheduler_type="constant", optim="adamw_torch", weight_decay=PLAN["weight_decay"],
        eval_strategy="epoch", save_strategy="steps", save_steps=10, save_total_limit=2,
        logging_strategy="steps", logging_steps=1, prediction_loss_only=True,
        remove_unused_columns=False, label_names=["labels"], dataloader_pin_memory=device == "cuda",
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data, eval_dataset=val_data,
                      data_collator=partial(padded_batch, pad_token_id=tokenizer.pad_token_id))
    checkpoint = last_complete_checkpoint(checkpoints)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    status = "interrupted"
    try:
        trainer.train(resume_from_checkpoint=checkpoint)
        val_loss = [log["eval_loss"] for log in trainer.state.log_history if "eval_loss" in log][-1]
        stats = {"trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                 "optimizer_steps": trainer.state.global_step, "truncated_targets": truncated,
                 "last_validation_loss": val_loss,
                 "peak_gpu_vram_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}
        # Publish the adapter directory only after weights, logs and stats have all been written.
        with TemporaryDirectory(dir=dest) as temp:
            ready = Path(temp) / "adapter"
            model.save_pretrained(ready, safe_serialization=True)
            save_json(ready / "training_log.json", trainer.state.log_history)
            save_json(ready / "training_stats.json", stats)
            ready.rename(dest / "adapter")
        status = "complete"
    finally:
        attempt = {"start_utc": started_at, "resume_checkpoint": Path(checkpoint).name if checkpoint else None,
                   "status": status, "elapsed_seconds": time.perf_counter() - started}
        with (dest / "training_attempts.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(attempt) + "\n")
    model.config.use_cache = True
    return model


def predictions(model, tokenizer, test, tag, dest):
    import torch
    from transformers import GenerationConfig

    path = dest / f"{tag}.predictions.jsonl"
    if path.exists():
        cached = read_jsonl(path)
        if [r["prompt_id"] for r in cached] != [r["id"] for r in test]:
            raise ValueError(f"Unexpected test IDs in {path}")
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
            for row, response in zip(batch, tokenizer.batch_decode(generated, skip_special_tokens=True)):
                response = response.strip()
                extracted, correct = score(response, row["reference_answer"])
                rows.append({"prompt_id": row["id"], "prompt": row["question"], "response": response,
                             "reference_answer": row["reference_answer"], "extracted_answer": extracted,
                             "is_correct": correct})
            print(f"{tag} | {len(rows)}/{len(test)}", flush=True)
    save_jsonl(path, rows)
    return rows


def run(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    folder = args.output / args.stage
    dest = folder / "A123"
    teacher_path = folder / "teacher_A.jsonl"
    sources = read_split(args.data, "train", args.stage)
    rows = teacher_set(teacher_path, sources)
    validation = read_split(args.data, "validation", args.stage)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    device = "cpu" if device == "auto" else device
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; run with --device cpu")
    torch.set_num_threads(args.threads)
    set_seed(PLAN["seed"])
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    manifest = {"stage": args.stage, "sft": PLAN, "device": device,
                "processor": platform.processor(), "platform": platform.platform(), "python": sys.version.split()[0],
                "gpu": torch.cuda.get_device_name() if device == "cuda" else None, "dtype": str(dtype),
                "threads": args.threads, "teacher_jsonl_sha256": hashlib.sha256(teacher_path.read_bytes()).hexdigest(),
                "source_splits": CONFIG["data"], "versions": {p: version(p) for p in
                ("torch", "transformers", "peft", "accelerate", "huggingface-hub", "safetensors")}}
    manifest_path = dest / "run_config.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Inputs, software or device changed. Select a different --output root for this run.")
    save_json(manifest_path, manifest)
    if (dest / "notes.md").is_file() and (dest / "metrics.json").is_file():
        print((dest / "notes.md").read_text(encoding="utf-8"))
        return

    tokenizer = AutoTokenizer.from_pretrained(PLAN["student"], revision=PLAN["revision"])
    base = AutoModelForCausalLM.from_pretrained(PLAN["student"], revision=PLAN["revision"],
                                                torch_dtype=dtype, use_safetensors=True).to(device)
    if (dest / "adapter").is_dir():
        model = PeftModel.from_pretrained(base, dest / "adapter")
    else:
        model = train(base, tokenizer, rows, validation, dest, device)
    test = read_split(args.data, "test", args.stage)
    metrics = {}
    for name in ("baseline", "student"):
        results = predictions(model, tokenizer, test, name, dest)
        correct = sum(r["is_correct"] for r in results)
        invalid = sum(r["extracted_answer"] is None for r in results)
        metrics[name] = {"n": len(results), "correct": correct, "accuracy": correct / len(results),
                         "invalid": invalid, "invalid_format_rate": invalid / len(results)}
    metrics["delta_pp"] = 100 * (metrics["student"]["accuracy"] - metrics["baseline"]["accuracy"])
    metrics["teacher_A"] = quality(rows)
    metrics["sft"] = read_json(dest / "adapter" / "training_stats.json")
    metrics["sft"]["recorded_seconds"] = sum(a["elapsed_seconds"] for a in read_jsonl(dest / "training_attempts.jsonl"))
    save_json(dest / "metrics.json", metrics)
    b, s, t = metrics["baseline"], metrics["student"], metrics["teacher_A"]
    notes = (f"# Эксперимент A123 · {args.stage}\n\n"
             f"Teacher A ({CONFIG['teachers']['A']['model']}, Q4_K_M) сгенерировал решения для {len(rows)} "
             f"англоязычных обучающих задач. Верных финальных ответов — {t['correct']} из {t['n']}, "
             f"ответов с нарушением формата — {t['invalid']}. Все решения вошли в обучение.\n\n"
             f"Исходная Student {PLAN['student']} прошла {PLAN['epochs']} эпохи LoRA SFT с seed {PLAN['seed']}, "
             f"rank {PLAN['rank']}, alpha {PLAN['alpha']} и эффективным батчем {PLAN['effective_batch_size']}. "
             f"Loss учитывал токены ответа и EOS. На {len(validation)} валидационных задачах целевым текстом "
             f"служила строка с эталонным числом. Итоговый validation loss — {metrics['sft']['last_validation_loss']:.4f}. "
             f"Записанное время попыток SFT — {metrics['sft']['recorded_seconds']/60:.1f} мин, устройство — {device}.\n\n"
             f"На тех же {len(test)} тестовых задачах исходная модель решила {b['correct']} ({b['accuracy']:.1%}), "
             f"после дообучения — {s['correct']} ({s['accuracy']:.1%}). Изменение — {metrics['delta_pp']:+.1f} п. п. "
             f"Парсер отклонил {b['invalid']} и {s['invalid']} ответов соответственно. Полные ответы обеих моделей "
             f"сохранены для проверки. Для сравнения влияния Teacher нужны остальные модели и второй seed.\n")
    atomic_text(dest / "notes.md", notes)
    print(notes)


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
