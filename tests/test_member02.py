import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

from generate import prepare_sft_files, run_teacher
from protocol import CONFIG, DATA, chat_prompt, read_json, read_jsonl, read_split, score, user_prompt
from run_a123 import train_command
from train import encode_example, load_examples


class CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        prefix = "user:" + messages[0]["content"] + "|assistant:"
        return prefix if add_generation_prompt else prefix + messages[-1]["content"] + "!"

    def __call__(self, text, *, add_special_tokens, max_length, truncation, return_offsets_mapping):
        ids = list(map(ord, text[:max_length]))
        return {"input_ids": ids, "attention_mask": [1] * len(ids),
                "offset_mapping": [(i, i + 1) for i in range(len(ids))]}


class ProtocolTests(unittest.TestCase):
    @unittest.skipUnless(DATA.is_file(), "requires the original project_data.zip")
    def test_exact_original_splits(self):
        seen_ids, seen_questions = set(), set()
        for split, expected in (("train", 1000), ("validation", 150), ("test", 200)):
            rows = read_split(DATA, split, "full")
            self.assertEqual(len(rows), expected)
            self.assertEqual(read_split(DATA, split, "pilot"), rows[:CONFIG["pilot"][split]])
            for row in rows:
                question = " ".join(row["question"].casefold().split())
                self.assertNotIn(row["id"], seen_ids)
                self.assertNotIn(question, seen_questions)
                seen_ids.add(row["id"])
                seen_questions.add(question)

    def test_final_number(self):
        self.assertEqual(score("A calculation.\n#### 4,000.00", "4,000"), ("4000.00", True))
        self.assertEqual(score("#### -0.5", "-.50"), ("-0.5", True))
        self.assertEqual(score("#### 15", "16"), ("15", False))
        for bad in ("16", "#### 1,60", "#### 16 apples", "#### 16\nmore", "#### 2\n#### 16", ""):
            with self.subTest(bad=bad):
                self.assertEqual(score(bad, "16"), (None, False))

    def test_assistant_only_and_truncation(self):
        tokenizer = CharacterTokenizer()
        prompt = user_prompt("One plus two?")
        prefix = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                               tokenize=False, add_generation_prompt=True)
        text = "Add the numbers.\n#### 3"
        item = encode_example(tokenizer, prompt, text, 512)
        self.assertEqual(item["labels"][:len(prefix)], [-100] * len(prefix))
        self.assertEqual(item["labels"][len(prefix):], list(map(ord, text + "!")))
        clipped = encode_example(tokenizer, prompt, text, len(prefix) + 5)
        self.assertEqual(clipped["labels"][-5:], list(map(ord, "Add t")))
        with self.assertRaises(ValueError):
            encode_example(tokenizer, prompt, text, len(prefix))

    def test_resume_assembly_and_teacher_handoff(self):
        sources = [{"id": "fixture_0", "question": "First fixture?", "reference_answer": "123456"},
                   {"id": "fixture_1", "question": "Second fixture?", "reference_answer": "8"}]
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            args = Namespace(output=dest, stage="pilot", threads=4, gpu_layers=0, shard=(0, 2), assemble=False)
            model = Mock()
            model.tokenize.return_value = [1, 2]
            model.create_completion.return_value = {
                "choices": [{"text": "test output\n#### 8", "finish_reason": "stop"}],
                "usage": {"completion_tokens": 4}}
            loader = Mock(return_value=model)
            run_teacher(args, "A", sources, loader)
            model.tokenize.assert_called_once_with(chat_prompt(sources[0]["question"]).encode(), add_bos=False, special=True)
            self.assertNotIn(b"123456", model.tokenize.call_args.args[0])
            args.shard, args.assemble = (0, 1), True
            with self.assertRaises(FileNotFoundError):
                run_teacher(args, "A", sources, loader)
            args.assemble = False
            run_teacher(args, "A", sources, loader)
            self.assertEqual(model.create_completion.call_count, 2)
            args.assemble = True
            run_teacher(args, "A", sources, Mock(side_effect=AssertionError("Model loaded on resume")))
            out = dest / "pilot" / "teacher_A.jsonl"
            self.assertEqual([r["question"] for r in read_jsonl(out)], [r["question"] for r in sources])
            self.assertEqual(read_json(dest / "pilot" / "teacher_A.metrics.json")["accuracy"], 0.5)
            args.assemble = False
            run_teacher(args, "B", sources, loader)
            # C produces one empty response. All three SFT inputs must drop the same ID.
            answered = model.create_completion.return_value
            model.create_completion.side_effect = [answered, {"choices": [{"text": "", "finish_reason": "stop"}],
                                                            "usage": {"completion_tokens": 0}}]
            run_teacher(args, "C", sources, loader)
            aligned = prepare_sft_files(dest / "pilot", sources)
            self.assertEqual(aligned["excluded_empty_ids"], ["fixture_1"])
            for teacher in "ABC":
                sft = dest / "pilot" / "sft" / f"teacher_{teacher}.jsonl"
                self.assertEqual([r["prompt_id"] for r in read_jsonl(sft)], ["fixture_0"])
                self.assertEqual(len(load_examples(sft, CharacterTokenizer(), 512)), 1)
            with self.assertRaises(ValueError):
                prepare_sft_files(dest / "pilot", list(reversed(sources)))
            with self.assertRaises(ValueError):
                prepare_sft_files(dest / "pilot", sources[:1])
            args.threads = 8
            with self.assertRaises(ValueError):
                run_teacher(args, "A", sources, loader)

    def test_personal_run_uses_team_entry_point(self):
        command = train_command(Path("train.jsonl"), Path("val.jsonl"), Path("out"), "cpu")
        self.assertEqual(Path(command[1]).name, "train.py")
        for flag, expected in (("--seed", "123"), ("--model", "Qwen/Qwen2.5-0.5B-Instruct"),
                               ("--epochs", "2"), ("--batch-size", "2"), ("--gradient-accumulation-steps", "8")):
            self.assertEqual(command[command.index(flag) + 1], expected)
        self.assertIn("--cpu", command)


if __name__ == "__main__":
    unittest.main()
