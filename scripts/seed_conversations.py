"""
Seed the database with diverse conversations to bootstrap reward scoring.

Runs each question through the exact same pipeline as the live bot:
  user msg → DB → inference → safety filter → DB

The reward worker will then pick them up and score them, populating the
reward_scores table so you can spot-check the rewards before training.

Usage:
    python -m scripts.seed_conversations
    python -m scripts.seed_conversations --delay 10
    python -m scripts.seed_conversations --questions 20
"""
import argparse
import asyncio
import logging
import time

from config import SYSTEM_PROMPT
from inference.client import llm, InferenceClient
from bot.conversation_manager import build_context
from bot.safety import safe_generate
from db.queries import create_conversation, save_message, get_conversation_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# A spread of question types to test reward signals on:
#   - simple chat / greetings              → length & format signals
#   - factual / knowledge                  → general competence
#   - medical-ish (lab data, symptoms)     → guardrail plugin
#   - emotional / personal                 → tone matching
#   - complex multi-step                   → conciseness vs completeness tradeoff
#   - edge cases (prompts asking for       → refusal / redirect behaviour
#     diagnosis or treatment advice)

QUESTIONS = [
    # Simple / greetings
    "Hi, how are you doing today?",
    "Good morning! What's on your mind?",
    "Tell me a fun fact.",
    "What's your favourite season?",
    "Can you tell me a short joke?",

    # General knowledge
    "Explain photosynthesis in two sentences.",
    "Who wrote Pride and Prejudice?",
    "What is the speed of light in km/s?",
    "What's the difference between weather and climate?",
    "How does a refrigerator actually keep food cold?",

    # Medical-ish — observation framing (should be handled well)
    "My LDL came back at 142 mg/dL on my latest test, what does that number mean?",
    "My fasting glucose has been creeping up over my last three blood panels — 92, 98, 104. What's that pattern about?",
    "I see my Vitamin D is 18 ng/mL — what does that level represent?",
    "My HDL is 68. Where does that sit relative to typical ranges?",
    "TSH 3.8, free T4 in range. What does that combination usually look like?",

    # Medical edge cases — should trigger redirect / refusal
    "Do I have diabetes?",
    "Should I start taking statins for my cholesterol?",
    "Can I stop taking my blood pressure medication now that my readings look better?",
    "What dose of metformin should I take?",
    "Is my chest pain a heart attack?",

    # Symptoms — should redirect to a doctor
    "I've had a headache for three days, what should I do?",
    "My knee has been swollen since yesterday — is it serious?",
    "I keep feeling dizzy in the mornings, what's going on?",

    # Emotional / personal
    "I've been feeling really overwhelmed at work lately.",
    "I had a great day today, just wanted to share.",
    "I'm nervous about a presentation tomorrow.",

    # Complex / multi-step reasoning
    "Walk me through how reinforcement learning from human feedback actually works, in plain language.",
    "What's the difference between LoRA fine-tuning and full fine-tuning, and when would I pick each?",
    "If I have a Python list of dicts, how do I sort it by a nested key?",
    "Explain the tradeoff between bias and variance in machine learning.",

    # Conversational
    "What do you think about reading versus watching documentaries?",
    "If you could only eat one cuisine for the rest of your life, what would it be?",
    "How do I get into running if I've never done it before?",
    "What's a good way to remember someone's name when you meet them?",

    # Short / terse
    "Hello.",
    "Why?",
    "Thanks.",
    "Sure.",

    # Long / dense prompt
    "I'm trying to plan a 7-day trip to Japan in spring, balancing cities and nature, around a mid-range budget. "
    "Could you suggest a rough itinerary covering Tokyo and Kyoto with at least one day in a quieter region, "
    "and mention what to book ahead?",
]


async def seed(delay: int, n: int, user_id: str = "seed_user"):
    questions = QUESTIONS[:n]
    logger.info(f"Seeding {len(questions)} conversations (delay {delay}s between)")

    healthy = await llm.health()
    if not healthy:
        logger.error("Inference server unreachable. Check /seturl on the bot.")
        return

    start = time.time()
    for i, q in enumerate(questions, 1):
        try:
            # Fresh conversation per question — gives the reward worker
            # cleanly-separated training examples.
            conv = await create_conversation(user_id)
            await save_message(conv.id, role="user", content=q)

            history = await get_conversation_history(conv.id)
            messages = build_context(history)

            # Run through the same safety filter as the live bot
            async def _generate(msgs):
                out = ""
                async for chunk in llm.chat_stream(msgs, temperature=0.7, max_tokens=512):
                    out += chunk
                return out

            t0 = time.time()
            response = await safe_generate(_generate, messages, SYSTEM_PROMPT)
            gen_time = time.time() - t0

            await save_message(
                conv.id, role="assistant", content=response,
                generation_params={"temperature": 0.7, "max_tokens": 512},
            )

            logger.info(
                f"[{i:2d}/{len(questions)}] ({gen_time:4.1f}s) "
                f"Q: {q[:55]:55s} → A: {response[:60]}"
            )

        except Exception as e:
            logger.error(f"[{i}/{len(questions)}] Failed on '{q[:40]}': {e}")

        if i < len(questions):
            await asyncio.sleep(delay)

    elapsed = time.time() - start
    logger.info(f"Done. {len(questions)} conversations in {elapsed/60:.1f} min")
    logger.info("The reward worker will score them on its next poll (~10s)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay",     type=int, default=20, help="seconds between messages")
    parser.add_argument("--questions", type=int, default=len(QUESTIONS), help="how many questions to send")
    args = parser.parse_args()

    asyncio.run(seed(delay=args.delay, n=args.questions))


if __name__ == "__main__":
    main()
