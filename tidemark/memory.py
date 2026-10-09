import numpy as np

from .ledger import Ledger

EXTRACT = """Below are messages a user sent to an assistant on {date}.

List facts about the user's own life that a good assistant should remember for future conversations: where they live,
work or study, health, diet, family, pets, home, belongings, routines, habits, plans and constraints. Write each fact as
one short standalone sentence about "the user". Skip one-off requests, general questions and opinions about the topic.

Then add facts that the messages clearly imply about the user's situation even though they are not stated, the kind a
careful listener would conclude (for example, finding scorpions in their boots in relentless dry heat implies the user
lives in a hot desert region such as Arizona). Mark those as implied. Only add an implied fact if you are fairly sure.

Return an empty list if the messages say nothing lasting about the user.

Messages:
{messages}"""

SCHEMA = {
    "type": "object",
    "properties": {"facts": {"type": "array", "items": {
        "type": "object",
        "properties": {"fact": {"type": "string"}, "implied": {"type": "boolean"}},
        "required": ["fact", "implied"]}}},
    "required": ["facts"],
}

CONFLICT = """Two things we learned about the same user, at different times.

Earlier ({old_date}): {old}
Later ({new_date}): {new}

Given the later information, is the earlier statement most likely no longer true for this user today? Answer Yes or No."""


class Memory:
    def __init__(self, llm, embed, implied=True, invalidate=True, k=8, threshold=0.5):
        self.llm, self.embed = llm, embed
        self.implied, self.invalidate = implied, invalidate
        self.k, self.threshold = k, threshold
        self.ledger = Ledger()
        self.vecs = []
        self.checks = 0

    def extract(self, messages, date):
        text = "\n".join(f"- {m}" for m in messages)
        out = self.llm.chat([{"role": "user", "content": EXTRACT.format(date=date, messages=text)}],
                            max_tokens=1024, schema=SCHEMA)
        facts = [f for f in out["facts"] if f["fact"].strip()]
        if not self.implied:
            facts = [f for f in facts if not f["implied"]]
        return facts

    def write(self, messages, date):
        """Add one session's user messages to memory. Returns the new facts."""
        new = [self.ledger.add(f["fact"].strip(), date, f["implied"]) for f in self.extract(messages, date)]
        if not new:
            return []
        vecs = self.embed([f.text for f in new])
        if self.invalidate and self.vecs:
            old = np.array(self.vecs)
            for f, v in zip(new, vecs):
                live = [i for i in np.argsort(-(old @ v)) if self.ledger.facts[i].current][: self.k]
                for i in live:
                    o = self.ledger.facts[i]
                    if o.date >= f.date:
                        continue
                    self.checks += 1
                    p = self.llm.yes_prob(CONFLICT.format(old_date=o.date[:10], old=o.text, new_date=f.date[:10],
                                                          new=f.text))
                    if p > self.threshold:
                        self.ledger.supersede(o, f)
        self.vecs.extend(vecs)
        return new
