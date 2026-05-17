"""
Qwen3 chat template formatting.
Qwen3 uses ChatML format: <|im_start|>role\ncontent<|im_end|>
"""
from config import SYSTEM_PROMPT


def format_for_training(messages: list[dict], enable_thinking: bool = False) -> str:
    """Format a message list into Qwen3's ChatML template for GRPO training.

    Thinking mode is off by default — Qwen3's <think>...</think> blocks
    burn 200-400 tokens before producing any scorable answer, which causes
    truncation and noisy rewards during GRPO. Setting enable_thinking=False
    prefills an empty think block so the model jumps straight to the answer.
    """
    out = ""
    for msg in messages:
        out += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    if not enable_thinking:
        out += "<think>\n\n</think>\n\n"   # mirrors tokenizer.apply_chat_template(..., enable_thinking=False)
    return out


def build_messages(history: list[dict], system_prompt: str = SYSTEM_PROMPT) -> list[dict]:
    """Prepend the system prompt to a conversation history."""
    return [{"role": "system", "content": system_prompt}] + history
