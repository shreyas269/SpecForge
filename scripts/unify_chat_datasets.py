"""
Merge multiple arrow datasets with different columns into a unified format.

Handles ShareGPT and UltraChat datasets which have different schemas:
- ShareGPT:   ids, messages [{role, content}]
- UltraChat:  uuid, idx, conversations [{role, content}]

Output schema: id (str), idx (int), text (str)
The text column contains pre-formatted conversation strings with the chat template applied,
ready for use with is_preformatted=True.

Uses the same message validation and chat template logic from specforge.data.parse to ensure
the formatted text is identical to what preprocessing.py produces with is_preformatted=False.
"""

import argparse
import json
import warnings
from pathlib import Path

from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer

from specforge.data.parse import GeneralParser, HarmonyParser, Parser, ThinkingParser
from specforge.data.template import TEMPLATE_REGISTRY, ChatTemplate


def build_parser(tokenizer, template: ChatTemplate) -> Parser:
    """Build the appropriate parser based on chat template's parser_type.

    Mirrors the parser selection logic in preprocessing.py.
    """
    if template.parser_type == "general":
        return GeneralParser(tokenizer, template)
    elif template.parser_type == "thinking":
        return ThinkingParser(tokenizer, template)
    elif template.parser_type == "openai-harmony":
        return HarmonyParser(tokenizer, template)
    else:
        raise ValueError(f"Invalid parser type: {template.parser_type}")


def format_conversation(conversation: list[dict], parser: Parser) -> str:
    """Format a conversation using the same validation and template logic as the parser.

    Replicates the message validation, system prompt injection, and chat template
    application from GeneralParser.parse() (lines 80-148 in parse.py) without
    the unnecessary tokenize+decode roundtrip.
    """
    if not conversation:
        return ""

    messages = []

    if conversation[0]["role"] == "system":
        messages.append({"role": "system", "content": conversation[0]["content"]})
        conversation = conversation[1:]
    else:
        system_prompt = getattr(parser, "system_prompt", None)
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

    for j, sentence in enumerate(conversation):
        role = sentence["role"]
        if j == 0:
            if role != "user":
                warnings.warn(
                    f"Conversation must start with a 'user' role, but found '{role}'. Conversation truncated."
                )
                break
        else:
            prev_role = conversation[j - 1]["role"]
            if role == "tool" and prev_role not in ["assistant", "tool"]:
                warnings.warn(
                    f"A 'tool' message must follow an 'assistant' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                )
                break
            if role == "assistant" and prev_role not in ["user", "tool"]:
                warnings.warn(
                    f"An 'assistant' message must follow a 'user' or 'tool' message, but was preceded by '{prev_role}'. Conversation truncated."
                )
                break
        tool_calls = sentence.get("tool_calls")
        if isinstance(tool_calls, str):
            try:
                sentence["tool_calls"] = json.loads(tool_calls)
            except json.JSONDecodeError:
                warnings.warn(f"Failed to parse tool_calls JSON: {tool_calls}")
                break
        messages.append(sentence)

    try:
        text = parser.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
    except (ValueError, TypeError):
        chat_template = parser.chat_template
        parts = []
        bos_token = getattr(parser.tokenizer, "bos_token", None)
        user_header = chat_template.user_header or ""
        assistant_header = chat_template.assistant_header or ""
        end_of_turn = chat_template.end_of_turn_token or ""

        if bos_token:
            parts.append(bos_token)

        for msg in messages:
            if msg["role"] == "system":
                parts.append(msg["content"])
            elif msg["role"] == "user":
                parts.append(f"{user_header}{msg['content']}")
            elif msg["role"] == "assistant":
                parts.append(f"{assistant_header}{msg['content']}{end_of_turn}")
        text = "".join(parts)

    return text


def normalize_sharegpt(dataset: Dataset, parser: Parser, idx_offset: int = 0, num_proc: int = 1) -> Dataset:
    """Normalize ShareGPT dataset to unified schema.

    ShareGPT has: ids, messages [{role, content}]
    We map to:    id, idx, text (pre-formatted string)
    """

    def transform(example, index):
        convos = example.get("messages") or []
        messages = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in convos]
        text = format_conversation(messages, parser)

        return {
            "id": str(example.get("ids", "")),
            "idx": idx_offset + index,
            "text": text,
        }

    return dataset.map(transform, with_indices=True, remove_columns=dataset.column_names, num_proc=num_proc)


def normalize_ultrachat(dataset: Dataset, parser: Parser, idx_offset: int = 0, num_proc: int = 1) -> Dataset:
    """Normalize UltraChat dataset to unified schema.

    UltraChat has: uuid, idx, conversations [{role, content}]
    We map to:     id, idx, text (pre-formatted string)
    """

    def transform(example, index):
        convos = example.get("conversations") or []
        messages = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in convos]
        text = format_conversation(messages, parser)

        return {
            "id": str(example.get("uuid", "")),
            "idx": idx_offset + index,
            "text": text,
        }

    return dataset.map(transform, with_indices=True, remove_columns=dataset.column_names, num_proc=num_proc)


NORMALIZERS = {
    "sharegpt": normalize_sharegpt,
    "ultrachat": normalize_ultrachat,
}


def merge_datasets(dataset_configs: list[dict], parser: Parser, num_proc: int = 1) -> Dataset:
    """Merge multiple datasets with different schemas into one.

    Args:
        dataset_configs: list of dicts with keys:
            - path: str, path to arrow file on disk
            - format: str, one of "sharegpt" or "ultrachat"
        parser: Parser instance for formatting conversations
        num_proc: number of processes for parallel map
    """
    normalized = []
    idx_offset = 0

    for config in dataset_configs:
        path = config["path"]
        fmt = config["format"]

        print(f"Loading {path} (format: {fmt})...")
        ds = Dataset.from_file(path)
        print(f"  Loaded {len(ds)} examples with columns: {ds.column_names}")

        normalizer = NORMALIZERS[fmt]
        ds_norm = normalizer(ds, parser, idx_offset=idx_offset, num_proc=num_proc)
        idx_offset += len(ds_norm)

        normalized.append(ds_norm)
        print(f"  Normalized to columns: {ds_norm.column_names}")

    merged = concatenate_datasets(normalized)
    print(f"\nMerged dataset: {len(merged)} examples, columns: {merged.column_names}")
    return merged


def main():
    parser = argparse.ArgumentParser(description="Merge arrow datasets with different columns")
    parser.add_argument(
        "--datasets",
        nargs="+",
        required=True,
        help="Dataset specs as 'path:format' pairs, e.g. /data/sharegpt.arrow:sharegpt /data/ultrachat.arrow:ultrachat",
    )
    parser.add_argument("--output", required=True, help="Output path for merged dataset")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--chat-template",
        required=True,
        help=f"Chat template name from TEMPLATE_REGISTRY (e.g. llama3). Available: {TEMPLATE_REGISTRY.get_all_template_names()}",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=8,
        help="Number of processes for parallel processing (default: 8)",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    template = TEMPLATE_REGISTRY.get(args.chat_template)
    conv_parser = build_parser(tokenizer, template)

    dataset_configs = []
    for spec in args.datasets:
        parts = spec.rsplit(":", 1)
        if len(parts) != 2 or parts[1] not in NORMALIZERS:
            parser.error(f"Invalid dataset spec '{spec}'. Use 'path:format' where format is one of {list(NORMALIZERS)}")
        dataset_configs.append({"path": parts[0], "format": parts[1]})

    merged = merge_datasets(dataset_configs, conv_parser, num_proc=args.num_proc)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(output_path))
    print(f"\nSaved merged dataset to {output_path}")


if __name__ == "__main__":
    main()
