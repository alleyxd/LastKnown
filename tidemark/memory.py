import json
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .ledger import Ledger

EXTRACT = """Below are messages a user sent to an assistant on {date}.

List facts about the user's own life that a good assistant should remember for future conversations: where they live,
work or study, health, diet, family, pets, home, belongings, routines, habits, plans and constraints. Write each fact as
one short standalone sentence about "the user". Skip one-off requests, general questions and opinions about the topic.

Then add at most three facts that the messages clearly imply about the user's situation even though they are never
stated, the kind a careful listener would conclude (for example, finding scorpions in their boots in relentless dry
heat implies the user lives in a hot desert region such as Arizona). Mark only those as implied; anything the user
actually said is not implied. Only add an implied fact if you are fairly sure.

Give every fact one topic:
home (city, region or address where they live), climate (weather and surroundings where they live), housing (their
home itself and belongings), work (job, employer, income), school, family (partner, children, friends), pets, health,
diet, religion, transport (how they get around), routine (regular schedule), hobby, activity (what they are doing at
this moment), travel (trips and visits), request (what they are asking for, looking for or curious about right now;
not a lasting fact about their life).

Return an empty list if the messages say nothing lasting about the user.

Messages:
{messages}"""

TOPICS = ["home", "climate", "housing", "work", "school", "family", "pets", "health", "diet", "religion", "transport",
          "routine", "hobby", "activity", "travel", "request"]
# lasting facts are compared within their group; "activity" only holds for the latest session it was mentioned in;
# requests and trips accumulate and never overturn anything
GROUPS = {"home": "place", "climate": "place", "housing": "place"}
CHECKED = {"home", "climate", "housing", "work", "school", "family", "pets", "health", "diet", "religion", "transport",
           "routine", "hobby"}

SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "maxItems": 15, "items": {
        "type": "object",
        "properties": {"fact": {"type": "string"}, "topic": {"enum": TOPICS}, "implied": {"type": "boolean"}},
        "required": ["fact", "topic", "implied"]}}},
    "required": ["facts"],
}

CONFLICT = """Earlier ({old_date}) we noted: {old}
Later ({new_date}) we learned: {new}

Taking the later information into account, is the earlier note still true about the user today? Answer Yes or No."""

VERIFY = """Earlier ({old_date}) we noted: {old}
Later ({new_date}) we learned: {new}

Does the later note mean the earlier note is no longer true about the user today? Visits, trips, hotel stays, new
interests, unrelated topics and vague guesses do not overturn anything. Use common sense about what the later note
implies. Answer with JSON {{"no_longer_true": true}} or {{"no_longer_true": false}}."""


PREMISE = """Question from the user: {question}

Does this question assume or rely on the following being true? "{fact}" Answer Yes or No."""


def clip(text, n=2000):
    """Some users paste whole documents; keep the start, which is where they say what they want."""
    return text if len(text) <= n else text[:n] + " [...]"


class Memory:
    def __init__(self, llm, embed, implied=True, invalidate=True, k=5, threshold=0.5):
        self.llm, self.embed = llm, embed
        self.implied, self.invalidate = implied, invalidate
        self.k, self.threshold = k, threshold
        self.ledger = Ledger()
        self.vecs = []
        self.checks = self.verified = 0

    def extract(self, messages, date):
        text = "\n".join(f"- {clip(m)}" for m in messages)
        try:
            out = self.llm.chat([{"role": "user", "content": EXTRACT.format(date=date, messages=text)}],
                                max_tokens=2048, schema=SCHEMA)
        except json.JSONDecodeError:
            return []
        facts = [f for f in out["facts"] if f["fact"].strip()]
        if not self.implied:
            facts = [f for f in facts if not f["implied"]]
        return facts

    def write(self, messages, date):
        """Add one session's user messages to memory. Returns the new facts."""
        new = [self.ledger.add(f["fact"].strip(), date, f["implied"], f["topic"]) for f in self.extract(messages, date)]
        if not new:
            return []
        vecs = self.embed([f.text for f in new])
        if self.invalidate:
            if any(f.topic == "activity" for f in new):
                for o in self.ledger.current():
                    if o.topic == "activity" and o.date < date:
                        self.ledger.supersede(o, next(f for f in new if f.topic == "activity"), expired=True)
            pairs = []
            for f, v in zip(new, vecs):
                if f.topic not in CHECKED:
                    continue
                group = GROUPS.get(f.topic, f.topic)
                cands = [o for o in self.ledger.current() if o.date < f.date and o.topic in CHECKED]
                cands.sort(key=lambda o: -float(self.vecs[o.id] @ v))
                same = [o for o in cands if GROUPS.get(o.topic, o.topic) == group][: self.k]
                # topics are noisy, so also look at the closest facts filed elsewhere
                pairs += [(o, f) for o in same + [o for o in cands[:3] if o not in same]]
            self.checks += len(pairs)
            with ThreadPoolExecutor(16) as pool:
                probs = list(pool.map(self.llm.yes_prob, [self._fmt(CONFLICT, o, f) for o, f in pairs]))
                flagged = [pr for pr, p in zip(pairs, probs) if 1 - p > self.threshold]
                verdicts = list(pool.map(self._verify, flagged))
            self.verified += len(flagged)
            for (o, f), bad in zip(flagged, verdicts):
                if bad:
                    self.ledger.supersede(o, f)
        self.vecs.extend(vecs)
        return new

    @staticmethod
    def _fmt(tmpl, o, f):
        return tmpl.format(old_date=o.date[:10], old=o.text, new_date=f.date[:10], new=f.text)

    def _verify(self, pair):
        """Second look at a flagged pair, letting the model think it through."""
        text = self.llm.chat([{"role": "user", "content": self._fmt(VERIFY, *pair)}], max_tokens=3000, think=True)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
        m = re.search(r'"no_longer_true"\s*:\s*(true|false)', text)
        return bool(m) and m.group(1) == "true"

    def premise_notes(self, question, k=5, threshold=0.5):
        """Warnings for outdated facts that the question takes for granted."""
        old = self.ledger.contradicted()
        if not old:
            return []
        q = self.embed([question])[0]
        sims = np.array([self.vecs[f.id] for f in old]) @ q
        top = [old[i] for i in np.argsort(-sims)[:k]]
        with ThreadPoolExecutor(k) as pool:
            probs = list(pool.map(self.llm.yes_prob, [PREMISE.format(question=question, fact=f.text) for f in top]))
        notes = []
        for f, p in zip(top, probs):
            if p > threshold:
                new = self.ledger.facts[f.replaced_by]
                notes.append(f'The question assumes "{f.text}" but that is no longer true. Since {f.replaced_on[:10]}: '
                             f"{new.text}")
        return notes
