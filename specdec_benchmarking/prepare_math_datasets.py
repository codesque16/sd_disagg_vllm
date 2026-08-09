#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prepare GSM8K / MATH-500 / quality-prompt JSONL for vllm bench serve.

Writes CustomDataset JSONL rows:
  {"prompt": "...", "output_tokens": N, "answer": "...", "source": "...", "id": "..."}

Usage:
  python3 prepare_math_datasets.py
  python3 prepare_math_datasets.py --out-dir ./datasets --gsm8k-n 200 --math500-n 200
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

DEFAULT_OUTPUT_TOKENS = 1024

# 20 fixed prompts for garbage / acceptance sanity checks under different temps.
# Math items include a numeric/string gold answer for crude correctness checks.
QUALITY_PROMPTS: list[dict] = [
    {
        "id": "q01",
        "source": "quality",
        "prompt": (
            "Natalia sold clips to 48 of her friends in April, and then she sold "
            "half as many clips in May. How many clips did Natalia sell "
            "altogether in April and May? Show your work and put the final "
            "answer after the step-by-step solution, not before."
        ),
        "answer": "72",
    },
    {
        "id": "q02",
        "source": "quality",
        "prompt": (
            "A robe takes 2 bolts of blue fiber and half that much white fiber. "
            "How many bolts in total does it take? Show your work and put the "
            "final answer after the step-by-step solution, not before."
        ),
        "answer": "3",
    },
    {
        "id": "q03",
        "source": "quality",
        "prompt": (
            "Josh decides to try flipping a house. He buys a house for $80,000 "
            "and then puts in $50,000 in repairs. This increased the value of "
            "the house by 150%. How much profit did he make? Show your work "
            "and put the final answer after the step-by-step solution, not before."
        ),
        "answer": "70000",
    },
    {
        "id": "q04",
        "source": "quality",
        "prompt": (
            "There are 15 trees in the grove. Grove workers will plant trees "
            "in the grove today. After they are done, there will be 21 trees. "
            "How many trees did the grove workers plant today? Show your work "
            "and put the final answer after the step-by-step solution, not before."
        ),
        "answer": "6",
    },
    {
        "id": "q05",
        "source": "quality",
        "prompt": (
            "If there are 3 cars in the parking lot and 2 more cars arrive, "
            "how many cars are in the parking lot? Show your work and put the "
            "final answer after the step-by-step solution, not before."
        ),
        "answer": "5",
    },
    {
        "id": "q06",
        "source": "quality",
        "prompt": (
            "Leah had 32 chocolates and her sister had 42. If they ate 35, "
            "how many pieces do they have left in total? Show your work and "
            "put the final answer after the step-by-step solution, not before."
        ),
        "answer": "39",
    },
    {
        "id": "q07",
        "source": "quality",
        "prompt": (
            "Jason had 20 lollipops. He gave Denny some lollipops. Now Jason "
            "has 12 lollipops. How many lollipops did Jason give to Denny? "
            "Show your work and put the final answer after the step-by-step "
            "solution, not before."
        ),
        "answer": "8",
    },
    {
        "id": "q08",
        "source": "quality",
        "prompt": (
            "Shawn has five toys. For Christmas, he got two toys each from his "
            "mom and dad. How many toys does he have now? Show your work and "
            "put the final answer after the step-by-step solution, not before."
        ),
        "answer": "9",
    },
    {
        "id": "q09",
        "source": "quality",
        "prompt": (
            "There were nine computers in the server room. Five more computers "
            "were installed each day, from Monday to Thursday. How many "
            "computers are now in the server room? Show your work and put the "
            "final answer after the step-by-step solution, not before."
        ),
        "answer": "29",
    },
    {
        "id": "q10",
        "source": "quality",
        "prompt": (
            "Michael had 58 golf balls. On Tuesday, he lost 23 golf balls. On "
            "Wednesday, he lost 2 more. How many golf balls did he have at "
            "the end of Wednesday? Show your work and put the final answer "
            "after the step-by-step solution, not before."
        ),
        "answer": "33",
    },
    {
        "id": "q11",
        "source": "quality",
        "prompt": (
            "Olivia has $23. She bought five bagels for $3 each. How much money "
            "does she have left? Show your work and put the final answer after "
            "the step-by-step solution, not before."
        ),
        "answer": "8",
    },
    {
        "id": "q12",
        "source": "quality",
        "prompt": (
            "What is 17 multiplied by 19? Show your work and put the final "
            "answer after the step-by-step solution, not before."
        ),
        "answer": "323",
    },
    {
        "id": "q13",
        "source": "quality",
        "prompt": (
            "A store sells apples for $2 each and oranges for $3 each. If Maya "
            "buys 4 apples and 5 oranges, how much does she spend in total? "
            "Show your work and put the final answer after the step-by-step "
            "solution, not before."
        ),
        "answer": "23",
    },
    {
        "id": "q14",
        "source": "quality",
        "prompt": (
            "A train travels 60 miles per hour for 2.5 hours. How many miles "
            "does it travel? Show your work and put the final answer after "
            "the step-by-step solution, not before."
        ),
        "answer": "150",
    },
    {
        "id": "q15",
        "source": "quality",
        "prompt": (
            "Solve for x: 3x + 7 = 28. Show your work and put the final answer "
            "after the step-by-step solution, not before."
        ),
        "answer": "7",
    },
    {
        "id": "q16",
        "source": "quality",
        "prompt": (
            "Write a short paragraph (3-4 sentences) explaining what "
            "speculative decoding is in large language models."
        ),
        "answer": None,
    },
    {
        "id": "q17",
        "source": "quality",
        "prompt": (
            "List three differences between TCP and UDP. Use a short numbered list."
        ),
        "answer": None,
    },
    {
        "id": "q18",
        "source": "quality",
        "prompt": (
            "Summarize in two sentences why GPUs are used for neural network training."
        ),
        "answer": None,
    },
    {
        "id": "q19",
        "source": "quality",
        "prompt": (
            "Give a step-by-step plan to debug a CUDA out-of-memory error when "
            "serving an LLM. Keep it under 8 bullet points."
        ),
        "answer": None,
    },
    {
        "id": "q20",
        "source": "quality",
        "prompt": (
            "What is the capital of France? Answer with just the city name, "
            "then one sentence of context."
        ),
        "answer": "Paris",
    },
]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {path}")


def _gsm8k_answer(raw: str) -> str:
    # GSM8K answers end with "#### <number>"
    m = re.search(r"####\s*(.+)\s*$", raw.strip())
    return (m.group(1) if m else raw).replace(",", "").strip()


def _load_gsm8k_test() -> list[dict]:
    """Prefer HF datasets; fall back to OpenAI raw JSONL."""
    try:
        from datasets import load_dataset

        ds = load_dataset("openai/gsm8k", "main", split="test")
        return [{"question": ex["question"], "answer": ex["answer"]} for ex in ds]
    except Exception as e:
        print(f"HF datasets gsm8k load failed ({e}); using raw JSONL fallback")

    import urllib.request

    url = (
        "https://raw.githubusercontent.com/openai/grade-school-math/"
        "master/grade_school_math/data/test.jsonl"
    )
    cache = Path.home() / ".cache" / "specdec_bench" / "gsm8k_test.jsonl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    if not cache.exists():
        print(f"Downloading {url}")
        urllib.request.urlretrieve(url, cache)
    rows = []
    with cache.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def prepare_gsm8k(out_path: Path, n: int, output_tokens: int, seed: int) -> None:
    import random

    data = _load_gsm8k_test()
    rng = random.Random(seed)
    rng.shuffle(data)
    if n > 0:
        data = data[:n]
    rows = []
    for i, ex in enumerate(data):
        rows.append(
            {
                "id": f"gsm8k_{i}",
                "source": "gsm8k",
                "prompt": (
                    f"{ex['question'].strip()}\n\nShow your work and put the "
                    "final answer after the step-by-step solution, not before."
                ),
                "output_tokens": output_tokens,
                "answer": _gsm8k_answer(ex["answer"]),
            }
        )
    _write_jsonl(out_path, rows)


def prepare_math500(out_path: Path, n: int, output_tokens: int, seed: int) -> None:
    from datasets import load_dataset

    # HuggingFaceH4/MATH-500 is the common 500-problem subset.
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.shuffle(seed=seed)
    if n > 0:
        ds = ds.select(range(min(n, len(ds))))
    rows = []
    for i, ex in enumerate(ds):
        problem = ex.get("problem") or ex.get("question") or ""
        answer = ex.get("answer") or ""
        rows.append(
            {
                "id": f"math500_{i}",
                "source": "math500",
                "prompt": (
                    f"{problem.strip()}\n\nShow your work and put the final "
                    "answer after the step-by-step solution, not before."
                ),
                "output_tokens": output_tokens,
                "answer": str(answer).strip(),
            }
        )
    _write_jsonl(out_path, rows)


def prepare_quality(out_path: Path, output_tokens: int) -> None:
    rows = []
    for item in QUALITY_PROMPTS:
        row = {
            "id": item["id"],
            "source": item["source"],
            "prompt": item["prompt"],
            "output_tokens": output_tokens,
        }
        if item.get("answer") is not None:
            row["answer"] = item["answer"]
        rows.append(row)
    _write_jsonl(out_path, rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "bench_data",
    )
    p.add_argument("--gsm8k-n", type=int, default=200, help="0 = all test")
    p.add_argument("--math500-n", type=int, default=200, help="0 = all")
    p.add_argument("--output-tokens", type=int, default=DEFAULT_OUTPUT_TOKENS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-download", action="store_true", help="Only write quality_20")
    args = p.parse_args()

    prepare_quality(args.out_dir / "quality_20.jsonl", args.output_tokens)
    if args.skip_download:
        return
    prepare_gsm8k(
        args.out_dir / "gsm8k.jsonl", args.gsm8k_n, args.output_tokens, args.seed
    )
    prepare_math500(
        args.out_dir / "math500.jsonl", args.math500_n, args.output_tokens, args.seed
    )


if __name__ == "__main__":
    main()
