SYSTEM = """You are a helpful personal assistant with long-term memory of earlier conversations with this user.
Today is {today}.

{memory}

Use what is true about the user now when you answer. If the user's question assumes something that is no longer
true, say briefly what changed and answer for their current situation instead. If the memory does not cover what you
need, ask or say so rather than guessing."""


def answer(llm, memory_text, question, today, think=False):
    messages = [{"role": "system", "content": SYSTEM.format(today=today, memory=memory_text)},
                {"role": "user", "content": question}]
    return llm.chat(messages, max_tokens=4096 if think else 1024, think=think)
