"""Two conversations months apart; the second quietly overturns the first.

LASTKNOWN_BASE_URL=http://localhost:8000/v1 LASTKNOWN_MODEL=qwen3-8b LASTKNOWN_THINKING_SWITCH=1 python examples/quickstart.py
"""
from lastknown import LLM, LocalEmbedder, Memory

mem = Memory(LLM(), LocalEmbedder())

mem.write(["I've been in Portland for years now, the rain doesn't even bother me anymore.",
           "Can you suggest a good waterproof jacket for walking my dog?"], "2024-02-10")
mem.write(["Found another bark scorpion in my boot this morning. This dry heat drives them indoors.",
           "What should I keep on hand so the toddler and the dog stay safe?"], "2024-09-03")

print(mem.context())
print()
print(mem.ask("Since I live in Portland, what rainy-day activities should I plan this weekend?", today="2024-09-20"))

mem.save("user.json")   # Memory.load("user.json", llm, embed) picks it up next time
