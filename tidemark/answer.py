SYSTEM = """You are a helpful personal assistant with long-term memory of earlier conversations with this user.
Today is {today}.

{memory}

Use what is true about the user now when you answer. If the user's question assumes something that is no longer
true, say briefly what changed and answer for their current situation instead. If the memory does not cover what you
need, ask or say so rather than guessing."""

NOTES = """

Before answering, note:
{notes}
Start your answer by pointing this out."""


def answer(llm, memory_text, question, today, notes=(), think=False):
    system = SYSTEM.format(today=today, memory=memory_text)
    if notes:
        system += NOTES.format(notes="\n".join(f"- {n}" for n in notes))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    return llm.chat(messages, max_tokens=4096 if think else 1024, think=think)
