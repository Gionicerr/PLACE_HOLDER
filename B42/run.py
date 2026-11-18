"""Check the released Teacher B data or run the team's Student trainer with seed 42."""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

from protocol import ROOT, read_jsonl, read_split, save_jsonl, user_prompt


STUDENT = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
RELEASE = {
    "pilot": (100, "d51bbee9b6bc51d991b30e7d909ab541a08b81d640b69dd5079bc953ffe337f1"),
    "full": (1000, "df52c93bfbed4a93d4eec55b6a18c60713b025fa9146e615d0076c4a1df3ab8d"),
}


def checked_input(stage):
    folder = Path(__file__).resolve().parent
    path = folder / f"teacher_B_{stage}.jsonl"
    if stage == "pilot" and not path.exists():
        path = folder / "teacher_B_pilot_check.jsonl"
    if not path.is_file():
        raise ValueError(f"Missing {path.name}. Put the released JSONL in the B42 folder.")
    count, expected_hash = RELEASE[stage]
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"{path.name} differs from the release. Expected SHA-256 {expected_hash}.")
    rows = read_jsonl(path)
    if len(rows) != count:
        raise ValueError(f"Expected {count} rows in {path.name}.")
    for index, row in enumerate(rows):
        if row["prompt_id"] != f"train_{index:04d}" or not (row["prompt"].strip() and row["response"].strip()):
            raise ValueError(f"Unexpected ID or empty training text in {path.name}, line {index + 1}.")
    print(f"OK: {path.name}, {count} rows, SHA-256 {expected_hash}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("smoke", "pilot", "full"),
                        help="one update on pilot data, two epochs on pilot, or two epochs on full")
    parser.add_argument("--check", action="store_true", help="verify inputs and exit before training")
    parser.add_argument("--data", type=Path, help="team project_data.zip or split folder for validation loss")
    parser.add_argument("--cpu", action="store_true", help="force CPU training")
    parser.add_argument("--output-dir", type=Path, help="new or empty trainer output directory")
    args = parser.parse_args()
    stage = "pilot" if args.stage == "smoke" else args.stage
    try:
        train_file = checked_input(stage)
        validation = read_split(args.data, "validation", stage) if args.data else None
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"{error}\n")
    if validation is not None:
        print(f"OK: {len(validation)} original validation examples", flush=True)
    if args.check:
        return 0

    base = ROOT / "runs" / "B42"
    destination = args.output_dir or base / args.stage
    command = [sys.executable, str(ROOT / "train.py"),
               "--train-file", str(train_file), "--output-dir", str(destination),
               "--model", STUDENT, "--revision", REVISION, "--seed", "42"]
    if validation is not None:
        path = base / "inputs" / f"validation_{stage}.jsonl"
        save_jsonl(path, [{"prompt_id": row["id"], "prompt": user_prompt(row["question"]),
                           "response": "#### " + row["reference_answer"]} for row in validation])
        command.extend(["--validation-file", str(path)])
    if args.stage == "smoke":
        command.extend(["--max-steps", "1"])
    if args.cpu:
        command.append("--cpu")
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
