"""
Qwen3 chat template formatting.
Qwen3 uses ChatML format: <|im_start|>role\ncontent<|im_end|>
"""
from config import SYSTEM_PROMPT


def format_for_training(messages: list[dict]) -> str:
    """Format a message list into Qwen3's ChatML template for GRPO training."""
    out = ""
    for msg in messages:
        out += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"   # model generates from here
    return out


def build_messages(history: list[dict], system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Prepend the system prompt to a conversation history."""
    return [{"role": "system", "content": system_prompt}] + history
