from dataclasses import dataclass, field


@dataclass
class Fact:
    id: int
    text: str
    date: str
    implied: bool = False
    topic: str = ""
    replaced_by: int | None = None
    replaced_on: str | None = None
    sources: list = field(default_factory=list)

    @property
    def current(self):
        return self.replaced_by is None


class Ledger:
    """What we currently believe about the user, plus what used to be true."""

    def __init__(self):
        self.facts = []

    def add(self, text, date, implied=False, topic=""):
        f = Fact(len(self.facts), text, date, implied, topic)
        self.facts.append(f)
        return f

    def supersede(self, old, new):
        if old.current:
            old.replaced_by, old.replaced_on = new.id, new.date

    def current(self):
        return [f for f in self.facts if f.current]

    def render(self, history=True):
        lines = ["What is true about the user now (most recent first):"]
        for f in sorted(self.current(), key=lambda f: f.date, reverse=True):
            lines.append(f"- [{f.date[:10]}] {f.text}" + (" (inferred)" if f.implied else ""))
        old = [f for f in self.facts if not f.current]
        if history and old:
            lines += ["", "No longer true (kept for history):"]
            for f in sorted(old, key=lambda f: f.date):
                new = self.facts[f.replaced_by]
                lines.append(f"- [{f.date[:10]}] {f.text} -> since {f.replaced_on[:10]}: {new.text}")
        return "\n".join(lines)

    def to_json(self):
        return [vars(f) for f in self.facts]
