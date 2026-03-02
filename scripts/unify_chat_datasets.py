"""
Merge multiple arrow datasets with different columns into a unified format.

Handles ShareGPT and UltraChat datasets which have different schemas:
- ShareGPT:   ids, messages [{role, content}]
- UltraChat:  uuid, idx, conversations [{role, content}]

Output schema: id (str), idx (int), text (str)
The text column contains pre-formatted conversation strings with the chat template applied,
ready for use with is_preformatted=True.
"""

import argparse
from pathlib import Path

from datasets import Dataset, concatenate_datasets, load_from_disk
from transformers import AutoTokenizer


def normalize_sharegpt(dataset: Dataset, tokenizer, idx_offset: int = 0) -> Dataset:
    """Normalize ShareGPT dataset to unified schema.

    ShareGPT has: ids, messages [{role, content}]
    We map to:    id, idx, text (pre-formatted string)
    """

    def transform(example, index):
        convos = example.get("messages") or []
        messages = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in convos]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

        return {
            "id": str(example.get("ids", "")),
            "idx": idx_offset + index,
            "text": text,
        }

    return dataset.map(transform, with_indices=True, remove_columns=dataset.column_names)


def normalize_ultrachat(dataset: Dataset, tokenizer, idx_offset: int = 0) -> Dataset:
    """Normalize UltraChat dataset to unified schema.

    UltraChat has: uuid, idx, conversations [{role, content}]
    We map to:     id, idx, text (pre-formatted string)
    """

    def transform(example, index):
        convos = example.get("conversations") or []
        messages = [{"role": m.get("role", ""), "content": m.get("content", "")} for m in convos]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

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


def merge_datasets(dataset_configs: list[dict], tokenizer) -> Dataset:
    """Merge multiple datasets with different schemas into one.

    Args:
        dataset_configs: list of dicts with keys:
            - path: str, path to arrow dataset on disk
            - format: str, one of "sharegpt" or "ultrachat"
        tokenizer: tokenizer with chat template for formatting conversations
    """
    normalized = []
    idx_offset = 0

    for config in dataset_configs:
        path = config["path"]
        fmt = config["format"]

        print(f"Loading {path} (format: {fmt})...")
        ds = load_from_disk(path)
        print(f"  Loaded {len(ds)} examples with columns: {ds.column_names}")

        normalizer = NORMALIZERS[fmt]
        ds_norm = normalizer(ds, tokenizer, idx_offset=idx_offset)
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
        help="Dataset specs as 'path:format' pairs, e.g. /data/sharegpt:sharegpt /data/ultrachat:ultrachat",
    )
    parser.add_argument("--output", required=True, help="Output path for merged dataset")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Tokenizer name or path (e.g. meta-llama/Llama-3.1-8B-Instruct)",
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    dataset_configs = []
    for spec in args.datasets:
        parts = spec.rsplit(":", 1)
        if len(parts) != 2 or parts[1] not in NORMALIZERS:
            parser.error(f"Invalid dataset spec '{spec}'. Use 'path:format' where format is one of {list(NORMALIZERS)}")
        dataset_configs.append({"path": parts[0], "format": parts[1]})

    merged = merge_datasets(dataset_configs, tokenizer)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.save_to_disk(str(output_path))
    print(f"\nSaved merged dataset to {output_path}")


if __name__ == "__main__":
    main()
