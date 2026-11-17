import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import Mock

from generate import run_teacher
from protocol import CONFIG, DATA, chat_prompt, read_json, read_jsonl, read_split, score
from run_a123 import encode_example, last_complete_checkpoint, teacher_set


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
        tokenizer = Mock(eos_token_id=99)
        tokenizer.encode.side_effect = lambda s, **kw: [ord(c) for c in s]
        prompt = chat_prompt("One plus two?")
        text = "Add the two numbers.\n#### 3"
        item, cut = encode_example(tokenizer, "One plus two?", text, len(prompt) + 8)
        self.assertTrue(cut)
        self.assertEqual(item["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(item["labels"][-7:], [ord(c) for c in "#### 3"] + [99])
        self.assertEqual(len(item["input_ids"]), len(item["labels"]))
        with self.assertRaises(ValueError):
            encode_example(tokenizer, "One plus two?", text, len(prompt))

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
            self.assertEqual(teacher_set(out, sources), read_jsonl(out))
            self.assertEqual(read_json(dest / "pilot" / "teacher_A.metrics.json")["accuracy"], 0.5)
            with self.assertRaises(ValueError):
                teacher_set(out, list(reversed(sources)))
            with self.assertRaises(ValueError):
                teacher_set(out, sources[:1])
            args.threads = 8
            with self.assertRaises(ValueError):
                run_teacher(args, "A", sources, loader)

    def test_incomplete_checkpoint_is_not_resumed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older, newer = root / "checkpoint-10", root / "checkpoint-20"
            older.mkdir()
            newer.mkdir()
            for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                         "adapter_model.safetensors", "adapter_config.json"):
                (older / name).touch()
            (newer / "trainer_state.json").touch()
            self.assertEqual(last_complete_checkpoint(root), str(older))


if __name__ == "__main__":
    unittest.main()
