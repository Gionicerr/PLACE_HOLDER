"""Generate the three Teacher JSONL files on the exact same train prompts."""

import argparse
import hashlib
import os
import math
import statistics
import time
from importlib.metadata import version
from pathlib import Path

from protocol import CONFIG, DATA, chat_prompt, read_json, read_jsonl, read_split, save_json, save_jsonl, score, user_prompt


def settings(teacher, threads, gpu_layers):
    model = CONFIG["teachers"][teacher]
    return {**CONFIG["generation"], "model": model["model"],
            "gguf_repo": model["model"] + "-GGUF", "revision": model["gguf_revision"],
            "files": model["files"], "sha256": model["sha256"], "threads": threads, "gpu_layers": gpu_layers}


def load_teacher(cfg):
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama, llama_supports_gpu_offload

    if version("llama-cpp-python") != cfg["backend_version"]:
        raise ValueError(f"Install llama-cpp-python=={cfg['backend_version']} for this protocol")
    if cfg["gpu_layers"] and not llama_supports_gpu_offload():
        raise RuntimeError("GPU offload requires a GPU build of llama-cpp-python")
    directory = os.environ.get("QWEN_GGUF_DIR")
    paths = []
    for name, expected in zip(cfg["files"], cfg["sha256"], strict=True):
        path = (Path(directory) / name if directory else
                Path(hf_hub_download(cfg["gguf_repo"], name, revision=cfg["revision"])))
        with path.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                raise ValueError(f"Model file checksum mismatch: {name}")
        paths.append(str(path))
    # llama.cpp opens the other GGUF shards beside the first one.
    return Llama(model_path=paths[0], n_ctx=cfg["context_length"], n_threads=cfg["threads"],
                 n_threads_batch=cfg["threads"], n_gpu_layers=cfg["gpu_layers"], verbose=False)


def generate_one(model, cfg, row, teacher):
    tokens = model.tokenize(chat_prompt(row["question"]).encode(), add_bos=False, special=True)
    if len(tokens) + cfg["max_new_tokens"] > cfg["context_length"]:
        raise ValueError(f"{row['id']} exceeds the Teacher context window")
    started = time.perf_counter()
    result = model.create_completion(
        prompt=tokens, max_tokens=cfg["max_new_tokens"], temperature=cfg["temperature"],
        top_k=cfg["top_k"], top_p=cfg["top_p"], repeat_penalty=cfg["repeat_penalty"],
        seed=cfg["generation_seed"], stop=["<|im_end|>", "<|endoftext|>"],
    )
    seconds = time.perf_counter() - started
    choice = result["choices"][0]
    response = choice["text"].strip()
    extracted, correct = score(response, row["reference_answer"])
    return {"prompt_id": row["id"], "question": row["question"], "prompt": user_prompt(row["question"]), "teacher_id": teacher,
            "response": response, "reference_answer": row["reference_answer"],
            "generation_config": cfg, "extracted_answer": extracted, "is_correct": correct,
            "response_tokens": result["usage"]["completion_tokens"],
            "finish_reason": choice["finish_reason"], "generation_seconds": seconds}


def validate_record(record, source, teacher, cfg):
    expected = (source["id"], source["question"], user_prompt(source["question"]), source["reference_answer"], teacher, cfg)
    actual = tuple(record[k] for k in ("prompt_id", "question", "prompt", "reference_answer", "teacher_id", "generation_config"))
    if actual != expected:
        raise ValueError(f"{source['id']}: cached row belongs to a different input or run")
    extracted, correct = score(record["response"], source["reference_answer"])
    if (record["extracted_answer"], record["is_correct"]) != (extracted, correct):
        raise ValueError(f"{source['id']}: cached score disagrees with the response")
    if record["response_tokens"] < 0 or record["generation_seconds"] < 0:
        raise ValueError(f"{source['id']}: invalid generation statistics")


def quality(rows):
    lengths = sorted(r["response_tokens"] for r in rows)
    count = len(rows)
    correct = sum(r["is_correct"] for r in rows)
    invalid = sum(r["extracted_answer"] is None for r in rows)
    return {"n": count, "correct": correct, "accuracy": correct / count,
            "invalid": invalid, "invalid_format_rate": invalid / count,
            "length_limited": sum(r["finish_reason"] == "length" for r in rows),
            "response_tokens": {"mean": statistics.mean(lengths), "p50": statistics.median(lengths),
                                "p95": lengths[math.ceil(.95 * count) - 1],
                                "min": lengths[0], "max": lengths[-1]},
            "summed_generation_seconds": sum(r["generation_seconds"] for r in rows)}


def run_teacher(args, teacher, sources, model_loader=load_teacher):
    folder = args.output / args.stage
    parts = args.output / "rows" / teacher
    cfg = settings(teacher, args.threads, args.gpu_layers)
    shard, nshards = args.shard
    model = None
    try:
        for index in range(shard, len(sources), nshards):
            source = sources[index]
            dest = parts / (source["id"] + ".json")
            if dest.exists():
                validate_record(read_json(dest), source, teacher, cfg)
                continue
            if args.assemble:
                raise FileNotFoundError(f"Missing {dest}. Finish the Teacher run before assembly.")
            if model is None:
                model = model_loader(cfg)
            record = generate_one(model, cfg, source, teacher)
            save_json(dest, record)
            if index == shard or ((index - shard) // nshards + 1) % 10 == 0:
                print(f"Teacher {teacher} | {args.stage} | {source['id']}", flush=True)
    finally:
        if model is not None:
            model.close()
    if nshards == 1:
        rows = [read_json(parts / (source["id"] + ".json")) for source in sources]
        for record, source in zip(rows, sources):
            validate_record(record, source, teacher, cfg)
        metrics = {"teacher_id": teacher, "stage": args.stage, "train_sha256": CONFIG["data"]["train"]["sha256"],
                   "generation_config": cfg, **quality(rows)}
        save_jsonl(folder / f"teacher_{teacher}.jsonl", rows)
        save_json(folder / f"teacher_{teacher}.metrics.json", metrics)
        print(f"Teacher {teacher}: {metrics['correct']}/{metrics['n']} correct, {metrics['invalid']} invalid", flush=True)



def prepare_sft_files(folder, sources):
    """Use the same prompt IDs for all Teachers; the team trainer requires nonempty responses."""
    datasets = {}
    execution = None
    for teacher in CONFIG["teachers"]:
        rows = read_jsonl(folder / f"teacher_{teacher}.jsonl")
        if len(rows) != len(sources):
            raise ValueError(f"Teacher {teacher}: incomplete or extra train rows")
        backend = rows[0]["generation_config"]
        runtime = (backend["threads"], backend["gpu_layers"])
        if execution is not None and runtime != execution:
            raise ValueError("Teacher execution settings differ")
        execution = runtime
        cfg = settings(teacher, *runtime)
        for row, source in zip(rows, sources):
            validate_record(row, source, teacher, cfg)
        datasets[teacher] = rows

    keep = [i for i in range(len(sources)) if all(rows[i]["response"].strip() for rows in datasets.values())]
    if not keep:
        raise ValueError("No common nonempty training responses")
    keep_ids = {sources[i]["id"] for i in keep}
    out = folder / "sft"
    for teacher, rows in datasets.items():
        save_jsonl(out / f"teacher_{teacher}.jsonl",
                   [{key: rows[i][key] for key in ("prompt_id", "prompt", "response")} for i in keep])
    manifest = {"n_original": len(sources), "n_sft": len(keep),
                "included_ids": [sources[i]["id"] for i in keep],
                "excluded_empty_ids": [r["id"] for r in sources if r["id"] not in keep_ids]}
    save_json(out / "alignment.json", manifest)
    return manifest


def shard_arg(value):
    try:
        index, count = (int(part) for part in value.split("/"))
        if 0 <= index < count:
            return index, count
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("Specify a zero-based shard such as 0/4")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA, help="The supplied ZIP or a directory of original splits")
    parser.add_argument("--output", type=Path, default=Path("runs"))
    parser.add_argument("--stage", choices=["pilot", "full"], default="pilot")
    parser.add_argument("--teacher", choices=list(CONFIG["teachers"]), action="append", help="Repeat to select several. Default: A, B, C")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--gpu-layers", type=int, default=0, help="Use -1 for full GPU offload with a CUDA build")
    parser.add_argument("--shard", type=shard_arg, default=(0, 1), metavar="INDEX/COUNT")
    parser.add_argument("--assemble", action="store_true", help="Validate completed rows and export JSONL without loading a model")
    args = parser.parse_args()
    if args.threads < 1 or args.gpu_layers < -1 or (args.assemble and args.shard != (0, 1)):
        parser.error("Invalid threads, GPU layers or shard for assembly")
    sources = read_split(args.data, "train", args.stage)
    for teacher in args.teacher or CONFIG["teachers"]:
        run_teacher(args, teacher, sources)
    folder = args.output / args.stage
    if args.shard == (0, 1) and all((folder / f"teacher_{t}.jsonl").exists() for t in CONFIG["teachers"]):
        prepare_sft_files(folder, sources)


if __name__ == "__main__":
    main()
