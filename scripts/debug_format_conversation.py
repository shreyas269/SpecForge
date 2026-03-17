"""
Debug script to visualize how a single gpt-oss conversation record gets formatted
into harmony prompt format.

Usage: python scripts/debug_format_conversation.py
"""

from specforge.data.parse import HarmonyParser
from specforge.data.template import TEMPLATE_REGISTRY
from transformers import AutoTokenizer

# --- Hardcode your record here ---
record = {
    "conversation_id": "0",
    "conversations": [
        {
            "role": "user",
            "content": "What is 2+2?",
            "thinking": None,
        },
        {
            "role": "assistant",
            "content": "The answer is 4.",
            "thinking": "This is a simple arithmetic question. 2+2=4.",
        },
        {
            "role": "user",
            "content": "And 3+3?",
            "thinking": None,
        },
        {
            "role": "assistant",
            "content": "6",
            "thinking": None,
        },
    ],
    "reasoning_effort": "low",
}
# --- End of record ---

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b", trust_remote_code=True)
template = TEMPLATE_REGISTRY.get("gpt-oss")
parser = HarmonyParser(tokenizer, template)

# Build the formatted text
reasoning_effort = record.get("reasoning_effort") or parser.default_reasoning_level
prompt_text = ""
prompt_text = parser.build_single_turn_prompt(prompt_text, "assistant_reasoning_effort", reasoning_effort)

for message in record["conversations"]:
    role = message["role"]
    content = message["content"]
    thinking = message.get("thinking")

    if role == "system":
        prompt_text = parser.build_single_turn_prompt(prompt_text, "system", content)
    elif role == "user":
        prompt_text = parser.build_single_turn_prompt(prompt_text, "user", content)
    elif role == "assistant":
        if thinking:
            prompt_text = parser.build_single_turn_prompt(prompt_text, "assistant_analysis", thinking)
        prompt_text = parser.build_single_turn_prompt(prompt_text, "assistant_final", content)

print("=" * 60)
print("FORMATTED TEXT")
print("=" * 60)
print(prompt_text)
print()

# Tokenize and show loss mask
input_ids, loss_mask = parser.parse(record["conversations"], max_length=2048)
tokens = tokenizer.convert_ids_to_tokens(input_ids)

print("=" * 60)
print("TOKEN-LEVEL VIEW (loss_mask=1 means trained)")
print("=" * 60)
for i, (tok, mask) in enumerate(zip(tokens, loss_mask)):
    marker = "*" if mask.item() == 1 else " "
    print(f"  {i:4d} [{marker}] {tok}")

print()
print(f"Total tokens: {len(input_ids)}")
print(f"Trained tokens: {loss_mask.sum().item()}")
