"""Hand-written fact pairs (not from any benchmark) for choosing the conflict-check wording.

python bench/conflict_dev.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tidemark import LLM  # noqa: E402

# (earlier, later, earlier is no longer true)
PAIRS = [
    ("The user lives in Denver.", "The user just signed a lease on an apartment in Miami and moved last week.", True),
    ("The user works as a nurse.", "The user has started their first semester of law school full time after quitting their job.", True),
    ("The user is vegetarian.", "The user has been trying out new steak recipes every weekend.", True),
    ("The user owns a cat named Milo.", "The user is still grieving after Milo passed away last month.", True),
    ("The user commutes by bike every day.", "The user broke their leg and will be on crutches for two months.", True),
    ("The user is single.", "The user is planning their wedding with their fiancee for next spring.", True),
    ("The user lives in a snowy mountain town.", "The user spends most evenings on the beach watching the surfers near their home.", True),
    ("The user drinks a lot of coffee every morning.", "The user's doctor told them to cut out caffeine completely, and they have.", True),
    ("The user rents a small studio apartment.", "The user is renovating the kitchen in the house they bought.", True),
    ("The user is a university student.", "The user graduated last spring and now works at a bank.", True),
    ("The user drives a gas car.", "The user sold their car and now gets around only by train.", True),
    ("The user works night shifts at a warehouse.", "The user's new office job runs nine to five, Monday to Friday.", True),
    ("The user lives in Toronto.", "The user enjoys hiking on weekends.", False),
    ("The user has a dog.", "The user is learning to play the guitar.", False),
    ("The user is vegetarian.", "The user is trying new tofu recipes.", False),
    ("The user works as a teacher.", "The user is planning a trip to Japan next summer.", False),
    ("The user lives in Chicago.", "The user is visiting their parents in Ohio this weekend.", False),
    ("The user is training for a marathon.", "The user bought new running shoes.", False),
    ("The user has two kids.", "The user's oldest child started kindergarten.", False),
    ("The user rents an apartment downtown.", "The user is looking for a quieter cafe near their place to work from.", False),
    ("The user enjoys cooking Italian food.", "The user is taking a Thai cooking class.", False),
    ("The user works remotely.", "The user is setting up a standing desk at home.", False),
    ("The user is allergic to peanuts.", "The user is looking for snack ideas for a road trip.", False),
    ("The user plays tennis on Saturdays.", "The user started a new job at a software company.", False),
    # realistic noise: request-style facts next to unrelated later facts
    ("The user is planning a road trip to the Grand Canyon.", "The user is looking for gift ideas for their sister.", False),
    ("The user is looking for budget-friendly meal prep ideas.", "The user is preparing for a job interview next week.", False),
    ("The user is interested in learning Spanish.", "The user is trying to fix a leaking kitchen faucet.", False),
    ("The user prefers quiet cafes for working.", "The user is comparing health insurance plans during open enrollment.", False),
    ("The user is concerned about drafts in their bedroom window.", "The user is experimenting with sourdough bread recipes.", False),
    ("The user is looking for podcast recommendations for long drives.", "The user takes photos of their car at national parks.", False),
    ("The user lives in Seattle.", "The user is planning a vacation to Tokyo next month.", False),
    ("The user lives in Seattle.", "The user is preparing for a trade show in Chicago.", False),
    ("The user works as an independent contractor.", "The user is looking for gift ideas for a coworker's birthday.", False),
    ("The user volunteers at a parish choir.", "The user is travelling a lot for work this month.", False),
    ("The user has a toddler.", "The user is redecorating the living room.", False),
    ("The user lives near the coast in Oregon.", "The user is shopping for a winter jacket.", False),
    ("The user lives in Austin.", "The user is looking for mid-range hotels in Honolulu near the beach.", False),
    ("The user lives in Denver.", "The user is likely living in a city or urban area.", False),
    ("The user is currently on a train for three more hours.", "The user is interested in concert venues downtown.", False),
    ("The user works as an independent contractor.", "The user uses Lightroom to edit photos.", False),
    ("The user just moved to London.", "The user is staying at a hotel in Tokyo this week.", False),
    ("The user lives in a dry desert climate.", "The user is likely in a region with a moderate climate.", False),
    ("The user has a dog and a toddler.", "The user is likely a young adult.", False),
    ("The user runs on the beach every morning.", "The user spends a lot of time unpacking boxes at their new place.", False),
    # implicit conflicts
    ("The user lives in a rainy coastal city.", "The user spends Sunday mornings sweeping red desert grit off the stucco and watering cacti.", True),
    ("The user works as a freelance designer with no employer.", "The user enrolled in their company's health and retirement plans at open enrollment.", True),
    ("The user lives at high altitude in the Andes.", "The user times their morning runs with the low tide on the beach near home.", True),
    ("The user sings in their church choir every Sunday.", "The user's new app chimes for each of the five daily prayers and shows the Qibla.", True),
]

PROMPTS = {
    "contradict": """Earlier ({od}) we noted: {o}
Later ({nd}) we learned: {n}

Does the later information show that the earlier note is no longer true about the user? Say Yes only if the later
information contradicts the earlier note or makes it impossible or very unlikely (a move, a job change, a new
religion, a different climate). Say No if the later note is about something else, adds detail, or fits alongside
it. Answer Yes or No.""",
    "both_true": """Two things we learned about the same user, at different times.

Earlier ({od}): {o}
Later ({nd}): {n}

Can both statements be true of this user at the same time, today? Answer Yes or No.""",
    "still_true": """Earlier ({od}) we noted: {o}
Later ({nd}) we learned: {n}

Taking the later information into account, is the earlier note still true about the user today? Answer Yes or No.""",
    "outdated": """Earlier ({od}) we noted: {o}
Later ({nd}) we learned: {n}

Does the later information make the earlier note outdated or false? Answer Yes or No.""",
}
# for these the answer Yes means the old fact survives
KEEP_IF_YES = {"both_true", "still_true"}


VERIFY = """Earlier ({od}) we noted: {o}
Later ({nd}) we learned: {n}

First name the one aspect of the user's life that both notes are about, in a few words (for example "home city",
"job", "religion", "diet"). If they are about different things, write "different". Then decide whether the later note
means the earlier one is no longer true. Visits, trips, hotel stays, guesses and new interests do not change where
someone lives or what they do."""

VERIFY_SCHEMA = {"type": "object", "properties": {"aspect": {"type": "string"}, "earlier_no_longer_true": {"type": "boolean"}},
                 "required": ["aspect", "earlier_no_longer_true"]}


def verify(llm, o, n):
    out = llm.chat([{"role": "user", "content": VERIFY.format(od="2023-02-01", o=o, nd="2024-05-01", n=n)}],
                   max_tokens=200, schema=VERIFY_SCHEMA)
    return out["earlier_no_longer_true"] and out["aspect"].strip().lower() != "different", out["aspect"]


def main():
    llm = LLM()
    for name, tmpl in PROMPTS.items():
        prompts = [tmpl.format(od="2023-02-01", o=o, nd="2024-05-01", n=n) for o, n, _ in PAIRS]
        with ThreadPoolExecutor(16) as pool:
            ps = list(pool.map(llm.yes_prob, prompts))
        stale = [1 - p if name in KEEP_IF_YES else p for p in ps]
        tp = sum(s > 0.5 and g for s, (_, _, g) in zip(stale, PAIRS))
        fp = sum(s > 0.5 and not g for s, (_, _, g) in zip(stale, PAIRS))
        pos = sum(g for *_, g in PAIRS)
        if name == "still_true":
            flagged = [i for i, x in enumerate(stale) if x > 0.9]
            with ThreadPoolExecutor(16) as pool:
                ver = dict(zip(flagged, pool.map(lambda i: verify(llm, PAIRS[i][0], PAIRS[i][1]), flagged)))
            tp2 = sum(ver[i][0] and PAIRS[i][2] for i in ver)
            fp2 = sum(ver[i][0] and not PAIRS[i][2] for i in ver)
            print(f"two-stage    caught {tp2}/{sum(g for *_, g in PAIRS)} conflicts, {fp2} false alarms; "
                  + "; ".join(f"{i}:{ver[i][1]}={ver[i][0]}" for i in ver))
        print(f"{name:12} caught {tp}/{pos} conflicts, {fp}/{len(PAIRS) - pos} false alarms  "
              + " ".join(f"{s:.2f}" for s in stale))


if __name__ == "__main__":
    main()
