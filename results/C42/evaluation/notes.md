# C42 — Evaluation results

Teacher: C
Seed: 42
Student: Qwen/Qwen2.5-0.5B-Instruct
Test examples: 200

## Baseline

- Correct: 0/200
- Accuracy: 0.00%
- Invalid format: 199
- Runtime: 186.99 sec

## Student

- Correct: 71/200
- Accuracy: 35.50%
- Invalid format: 46
- Runtime: 258.49 sec

## Difference

- Delta: +35.50 percentage points

## Files

- baseline.predictions.jsonl
- student.predictions.jsonl
- metrics.json

## Evaluation limitations and observations

Generation was limited to 192 new tokens.

- Baseline: 159 of 200 responses reached at least 192 tokens.
- Student C42: 19 of 200 responses reached at least 192 tokens.
- Baseline mean response length: approximately 175.0 tokens.
- Student C42 mean response length: approximately 109.6 tokens.

The baseline frequently failed to produce the required final answer line.
This is consistent with generation truncation, although exact generation
stop reasons were not recorded in the prediction files.

Therefore, the observed accuracy difference should be interpreted
with this generation-length limitation in mind. The agreed evaluation
protocol and original metrics were preserved.
