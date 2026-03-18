"""
Print statistics for a gpt-oss regenerated .jsonl dataset.

Usage:
    python scripts/dataset_stats.py --input /path/to/data.jsonl
    python scripts/dataset_stats.py --input /path/to/data.jsonl --tokenizer openai/gpt-oss-120b
"""

import argparse
import json
import sys
from collections import Counter

from transformers import AutoTokenizer
from tqdm import tqdm


def count_tokens(tokenizer, text):
    if not text:
        return 0
    return len(tokenizer.encode(text, add_special_tokens=False))


def main():
    parser = argparse.ArgumentParser(description="Print dataset statistics for a .jsonl file")
    parser.add_argument("--input", type=str, required=True, help="Path to .jsonl file")
    parser.add_argument("--tokenizer", type=str, default="openai/gpt-oss-120b", help="Tokenizer to use")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    num_records = 0
    num_errors = 0
    turn_counts = []
    user_turn_counts = []
    assistant_turn_counts = []
    record_token_counts = []
    thinking_token_counts = []
    response_token_counts = []
    reasoning_effort_counter = Counter()
    records_with_thinking = 0
    turns_with_thinking = 0
    total_assistant_turns = 0

    with open(args.input, "r") as f:
        for line in tqdm(f, desc="Processing"):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if record.get("status") == "error":
                num_errors += 1
                continue

            num_records += 1
            conversations = record.get("conversations", [])
            turn_counts.append(len(conversations))

            reasoning_effort = record.get("reasoning_effort")
            if reasoning_effort:
                reasoning_effort_counter[reasoning_effort] += 1

            record_tokens = 0
            user_turns = 0
            assistant_turns = 0
            has_thinking = False

            for msg in conversations:
                role = msg.get("role", "")
                content = msg.get("content", "")
                thinking = msg.get("thinking")

                content_tokens = count_tokens(tokenizer, content)
                record_tokens += content_tokens

                if role == "user":
                    user_turns += 1
                elif role == "assistant":
                    assistant_turns += 1
                    total_assistant_turns += 1
                    response_token_counts.append(content_tokens)

                    if thinking:
                        thinking_tokens = count_tokens(tokenizer, thinking)
                        thinking_token_counts.append(thinking_tokens)
                        record_tokens += thinking_tokens
                        turns_with_thinking += 1
                        has_thinking = True

            user_turn_counts.append(user_turns)
            assistant_turn_counts.append(assistant_turns)
            record_token_counts.append(record_tokens)
            if has_thinking:
                records_with_thinking += 1

    if num_records == 0:
        print("No valid records found.")
        sys.exit(1)

    def stats(values, name):
        if not values:
            return f"  {name}: (no data)"
        values.sort()
        total = sum(values)
        avg = total / len(values)
        p50 = values[len(values) // 2]
        p90 = values[int(len(values) * 0.9)]
        p99 = values[int(len(values) * 0.99)]
        return (
            f"  {name}:\n"
            f"    min={values[0]}, max={values[-1]}, mean={avg:.1f}\n"
            f"    p50={p50}, p90={p90}, p99={p99}, total={total}"
        )

    print()
    print("=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"  File: {args.input}")
    print(f"  Tokenizer: {args.tokenizer}")
    print(f"  Total records: {num_records}")
    if num_errors:
        print(f"  Error records (skipped): {num_errors}")
    print()

    print("--- Turns ---")
    print(stats(turn_counts, "Turns per record"))
    print(stats(user_turn_counts, "User turns per record"))
    print(stats(assistant_turn_counts, "Assistant turns per record"))
    print()

    print("--- Tokens (content only, no special tokens) ---")
    print(stats(record_token_counts, "Tokens per record"))
    print(stats(response_token_counts, "Tokens per assistant response"))
    print()

    print("--- Thinking ---")
    print(f"  Records with thinking: {records_with_thinking}/{num_records} ({records_with_thinking/num_records*100:.1f}%)")
    print(f"  Assistant turns with thinking: {turns_with_thinking}/{total_assistant_turns} ({turns_with_thinking/total_assistant_turns*100:.1f}%)" if total_assistant_turns else "  No assistant turns")
    if thinking_token_counts:
        print(stats(thinking_token_counts, "Tokens per thinking block"))
    print()

    if reasoning_effort_counter:
        print("--- Reasoning Effort ---")
        for level in ["low", "medium", "high"]:
            count = reasoning_effort_counter.get(level, 0)
            pct = count / num_records * 100
            print(f"  {level}: {count} ({pct:.1f}%)")
    print()


if __name__ == "__main__":
    main()
