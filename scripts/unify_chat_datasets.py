"""
Merge multiple arrow datasets with different columns into a unified format.

Handles ShareGPT and UltraChat datasets which have different schemas:
- ShareGPT:   ids, messages [{role, content}]
- UltraChat:  uuid, idx, conversations [{role, content}]

Output schema: id (str), idx (int), text (str)
The text column contains pre-formatted conversation strings with the chat template applied,
ready for use with is_preformatted=True.

Uses the same Parser logic from specforge.data.parse to ensure the formatted text
is identical to what preprocessing.py produces with is_preformatted=False.
"""

import argparse
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
    """Format a conversation using the same parser logic as preprocessing.py with preformatted=False.

    This reuses the parser's message validation, system prompt injection,
    and chat template application to produce identical output.
    """
    input_ids, _ = parser.parse(conversation, max_length=2**31)
    return parser.tokenizer.decode(input_ids, skip_special_tokens=False)


def normalize_sharegpt(dataset: Dataset, parser: Parser, idx_offset: int = 0) -> Dataset:
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

    return dataset.map(transform, with_indices=True, remove_columns=dataset.column_names)


def normalize_ultrachat(dataset: Dataset, parser: Parser, idx_offset: int = 0) -> Dataset:
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

    return dataset.map(transform, with_indices=True, remove_columns=dataset.column_names)


NORMALIZERS = {
    "sharegpt": normalize_sharegpt,
    "ultrachat": normalize_ultrachat,
}


def merge_datasets(dataset_configs: list[dict], parser: Parser) -> Dataset:
    """Merge multiple datasets with different schemas into one.

    Args:
        dataset_configs: list of dicts with keys:
            - path: str, path to arrow file on disk
            - format: str, one of "sharegpt" or "ultrachat"
        parser: Parser instance for formatting conversations
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
        ds_norm = normalizer(ds, parser, idx_offset=idx_offset)
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

    merged = merge_datasets(dataset_configs, conv_parser)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(output_path))
    print(f"\nSaved merged dataset to {output_path}")


if __name__ == "__main__":
    main()
