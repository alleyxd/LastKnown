import numpy as np

from tidemark import Ledger, Memory


class FakeLLM:
    """Returns canned facts per session and says two facts conflict when both mention a city."""

    def __init__(self, facts):
        self.facts = facts
        self.calls = 0

    def chat(self, messages, **kw):
        self.calls += 1
        return {"facts": self.facts.pop(0)}

    def yes_prob(self, prompt):
        earlier = prompt.split("Earlier")[1].split("Later")[0]
        later = prompt.split("Later")[1]
        return 0.9 if "lives in" in earlier and "lives in" in later else 0.1


def fake_embed(texts):
    vecs = []
    for t in texts:
        v = np.array([1.0 if "lives" in t else 0.0, 1.0 if "dog" in t else 0.0, 0.1])
        vecs.append(v / np.linalg.norm(v))
    return vecs


def test_later_fact_supersedes_earlier():
    llm = FakeLLM([
        [{"fact": "The user lives in Seattle.", "implied": False}, {"fact": "The user has a dog.", "implied": False}],
        [{"fact": "The user lives in Austin.", "implied": False}],
    ])
    mem = Memory(llm, fake_embed)
    mem.write(["I live in Seattle and walk my dog"], "2022-01-01 10:00")
    mem.write(["Just moved to Austin"], "2023-01-01 10:00")
    current = [f.text for f in mem.ledger.current()]
    assert "The user lives in Austin." in current
    assert "The user lives in Seattle." not in current
    assert "The user has a dog." in current
    text = mem.ledger.render()
    assert "No longer true" in text and "since 2023-01-01: The user lives in Austin." in text


def test_implied_facts_can_be_dropped():
    llm = FakeLLM([[{"fact": "The user lives in a desert region.", "implied": True}]])
    mem = Memory(llm, fake_embed, implied=False)
    assert mem.write(["scorpion in my boot again"], "2022-01-01") == []


def test_no_invalidation_keeps_everything():
    llm = FakeLLM([[{"fact": "The user lives in Seattle.", "implied": False}],
                   [{"fact": "The user lives in Austin.", "implied": False}]])
    mem = Memory(llm, fake_embed, invalidate=False)
    mem.write(["a"], "2022-01-01")
    mem.write(["b"], "2023-01-01")
    assert len(mem.ledger.current()) == 2 and mem.checks == 0


def test_render_orders_newest_first():
    led = Ledger()
    led.add("old", "2020-01-01")
    led.add("new", "2021-01-01")
    lines = led.render().splitlines()
    assert lines[1].endswith("new") and lines[2].endswith("old")
