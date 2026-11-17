"""Fine-tune a causal Student model on Teacher responses with SFT + LoRA.

Dependencies: torch 2.7.1, transformers 4.53.3, peft 0.16.0, accelerate 1.8.1.
Each line of the training or optional validation JSONL must contain nonempty
strings under "prompt" and "response". Additional fields are allowed.

Example:
    python train.py --train-file teacher_b.jsonl --seed 42 --output-dir runs/B42
"""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return number


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train-file", required=True, type=Path)
    parser.add_argument("--validation-file", type=Path, help="optional JSONL for validation loss")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--revision", default="main", help="Student checkpoint revision")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=positive_int, default=512)
    parser.add_argument("--epochs", type=positive_int, default=2)
    parser.add_argument("--max-steps", type=positive_int, help="override epochs for a smoke run")
    parser.add_argument("--batch-size", type=positive_int, default=2, help="per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=positive_int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=positive_int, default=8)
    parser.add_argument("--lora-alpha", type=positive_int, default=16)
    parser.add_argument("--bf16", action="store_true", help="enable BF16 mixed precision")
    parser.add_argument("--cpu", action="store_true", help="force CPU for a smoke run")
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be between 0 and 4294967295")
    if not 0 < args.learning_rate < float("inf"):
        parser.error("--learning-rate must be finite and positive")
    if args.output_dir.exists() and (
        not args.output_dir.is_dir() or any(args.output_dir.iterdir())
    ):
        parser.error("--output-dir must be a new or empty directory")
    return args


def encode_example(tokenizer, prompt, response, max_length):
    messages = [{"role": "user", "content": prompt}]
    prefix = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    chat = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": response}],
        tokenize=False,
        add_generation_prompt=False,
    )
    if not chat.startswith(prefix):
        raise ValueError("the chat template does not produce a consistent assistant prefix")

    encoded = tokenizer(
        chat,
        add_special_tokens=False,
        max_length=max_length,
        truncation=True,
        return_offsets_mapping=True,
    )
    # Keep only targets from the assistant segment, including its end-of-turn
    # token when it fits. Mask tokens that cross the prompt/response boundary.
    offsets = encoded.pop("offset_mapping")
    encoded["labels"] = [
        token if start >= len(prefix) and end > start else -100
        for token, (start, end) in zip(encoded["input_ids"], offsets)
    ]
    if not any(label != -100 for label in encoded["labels"]):
        raise ValueError("no assistant tokens remain; check --max-length and the prompt")
    return dict(encoded)


def load_examples(path, tokenizer, max_length):
    examples = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("expected a JSON object")
                prompt, response = row.get("prompt"), row.get("response")
                if not all(isinstance(value, str) and value.strip() for value in (prompt, response)):
                    raise ValueError("expected nonempty string fields 'prompt' and 'response'")
                examples.append(encode_example(tokenizer, prompt, response, max_length))
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
    if not examples:
        raise ValueError(f"{path}: no examples found")
    return examples


def main():
    args = parse_args()

    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Trainer,
        TrainingArguments,
        set_seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("assistant-only masking requires a fast tokenizer")
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("the tokenizer needs a padding or end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    train_data = load_examples(args.train_file, tokenizer, args.max_length)
    validation_data = None
    if args.validation_file:
        validation_data = load_examples(args.validation_file, tokenizer, args.max_length)

    set_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=args.revision)
    resolved_revision = getattr(model.config, "_commit_hash", None)
    lora = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear",
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        revision=resolved_revision or args.revision,
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        seed=args.seed,
        data_seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps is not None else -1,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        use_cpu=args.cpu,
        label_names=["labels"],
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
        logging_first_step=True,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100),
    )

    device = training_args.device
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    training = trainer.train()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    metrics = {
        **training.metrics,
        "train_samples": len(train_data),
        "train_time_seconds": time.perf_counter() - start,
        "peak_cuda_memory_allocated_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        ),
    }
    if validation_data is not None:
        metrics.update(trainer.evaluate())
        metrics["validation_samples"] = len(validation_data)
    trainer.save_model(str(args.output_dir / "adapter"))
    trainer.save_metrics("run", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        config = {
            **vars(args),
            "resolved_checkpoint_revision": resolved_revision,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "world_size": training_args.world_size,
            "versions": {
                package: version(package)
                for package in ("torch", "transformers", "peft", "accelerate")
            },
        }
        (args.output_dir / "run_config.json").write_text(
            json.dumps(config, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
