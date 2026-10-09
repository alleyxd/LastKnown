import json
import math
import os

from openai import OpenAI


class LLM:
    """Thin wrapper over an OpenAI-compatible endpoint (vLLM, llama.cpp server, ...)."""

    def __init__(self, base_url=None, model=None, api_key=None):
        self.client = OpenAI(base_url=base_url or os.environ.get("TIDEMARK_BASE_URL", "http://localhost:8000/v1"),
                             api_key=api_key or os.environ.get("TIDEMARK_API_KEY", "none"), timeout=600)
        self.model = model or os.environ.get("TIDEMARK_MODEL", "local")

    def chat(self, messages, max_tokens=1024, temperature=0.0, schema=None, think=False):
        extra = {"chat_template_kwargs": {"enable_thinking": think}}
        if schema:
            extra["structured_outputs"] = {"json": schema}
        r = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=max_tokens,
                                                temperature=temperature, extra_body=extra)
        text = r.choices[0].message.content or ""
        return json.loads(text) if schema else text.strip()

    def yes_prob(self, prompt):
        """P(Yes) from the first answer token, renormalised over Yes/No."""
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "user", "content": prompt}], max_tokens=1, temperature=0.0,
            logprobs=True, top_logprobs=20, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
        yes = no = 0.0
        for t in r.choices[0].logprobs.content[0].top_logprobs:
            word = t.token.strip().lower()
            if word == "yes":
                yes += math.exp(t.logprob)
            elif word == "no":
                no += math.exp(t.logprob)
        return yes / (yes + no) if yes + no else 0.5
