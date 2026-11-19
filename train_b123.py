"""Run member 04's B123 SFT with the shared trainer and released Teacher B data."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from B42.run import RELEASE, REVISION, STUDENT, checked_input
from protocol import CONFIG, ROOT, read_split, save_json, save_jsonl, user_prompt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="project_data.zip or split directory")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/B123/full"))
    args = parser.parse_args()

    teacher_file = checked_input("full")
    train = read_split(args.data, "train", "full")
    validation = read_split(args.data, "validation", "full")
    with teacher_file.open(encoding="utf-8") as source:
        for row, line in zip(train, source, strict=True):
            example = json.loads(line)
            if (example["prompt_id"], example["prompt"]) != (row["id"], user_prompt(row["question"])):
                raise ValueError(f"Teacher B example does not match train split: {row['id']}")

    with TemporaryDirectory(prefix="B123-") as temp:
        validation_file = Path(temp) / "validation.jsonl"
        save_jsonl(validation_file, [
            {"prompt_id": row["id"], "prompt": user_prompt(row["question"]),
             "response": "#### " + row["reference_answer"]}
            for row in validation
        ])
        subprocess.run([
            sys.executable, str(ROOT / "train.py"),
            "--train-file", str(teacher_file), "--validation-file", str(validation_file),
            "--output-dir", str(args.output_dir), "--model", STUDENT, "--revision", REVISION,
            "--seed", "123",
        ], check=True)
        save_json(args.output_dir / "inputs.json", {
            "teacher": "B", "teacher_sft_sha256": RELEASE["full"][1],
            "train_split": CONFIG["data"]["train"], "validation_split": CONFIG["data"]["validation"],
            "formatted_validation_sha256": hashlib.sha256(validation_file.read_bytes()).hexdigest(),
        })


if __name__ == "__main__":
    main()
