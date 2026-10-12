import json
import math
import os
import re

from openai import BadRequestError, OpenAI


def _env_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


class LLM:
    """Any OpenAI-compatible chat endpoint: OpenAI, vLLM, llama.cpp server, Ollama, LM Studio, SGLang, ...

    thinking_switch: send chat_template_kwargs.enable_thinking, for vLLM/SGLang serving models with a thinking toggle
    (Qwen3, DeepSeek). Leave off for everything else.
    """

    def __init__(self, base_url=None, model=None, api_key=None, thinking_switch=None):
        self.client = OpenAI(base_url=base_url or os.environ.get("TIDEMARK_BASE_URL", "http://localhost:8000/v1"),
                             api_key=api_key or os.environ.get("TIDEMARK_API_KEY")
                             or os.environ.get("OPENAI_API_KEY", "none"), timeout=600)
        self.model = model or os.environ.get("TIDEMARK_MODEL", "local")
        self.thinking_switch = _env_flag("TIDEMARK_THINKING_SWITCH") if thinking_switch is None else thinking_switch
        self.schema_ok = True      # flipped off if the server rejects json_schema
        self.logprobs_ok = True    # flipped off if the server returns no logprobs

    def _extra(self, think):
        return {"chat_template_kwargs": {"enable_thinking": think}} if self.thinking_switch else {}

    def chat(self, messages, max_tokens=1024, temperature=0.0, schema=None, think=False):
        kw = dict(model=self.model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                  extra_body=self._extra(think))
        if schema and self.schema_ok:
            try:
                r = self.client.chat.completions.create(
                    **kw, response_format={"type": "json_schema",
                                           "json_schema": {"name": "out", "schema": schema, "strict": False}})
                return json.loads(_strip(r.choices[0].message.content))
            except BadRequestError:
                self.schema_ok = False
        if schema:
            messages = messages[:-1] + [{"role": messages[-1]["role"], "content": messages[-1]["content"]
                                         + "\n\nReply with JSON only, matching this schema:\n" + json.dumps(schema)}]
            r = self.client.chat.completions.create(**{**kw, "messages": messages})
            return json.loads(_first_json(_strip(r.choices[0].message.content)))
        r = self.client.chat.completions.create(**kw)
        return _strip(r.choices[0].message.content)

    def yes_prob(self, prompt):
        """P(Yes) for a Yes/No question, read from the first answer token when the server gives logprobs."""
        messages = [{"role": "user", "content": prompt}]
        if self.logprobs_ok:
            try:
                r = self.client.chat.completions.create(model=self.model, messages=messages, max_tokens=1,
                                                        temperature=0.0, logprobs=True, top_logprobs=20,
                                                        extra_body=self._extra(False))
                lp = r.choices[0].logprobs
                if lp and lp.content:
                    yes = no = 0.0
                    for t in lp.content[0].top_logprobs:
                        word = t.token.strip().lower()
                        if word == "yes":
                            yes += math.exp(t.logprob)
                        elif word == "no":
                            no += math.exp(t.logprob)
                    return yes / (yes + no) if yes + no else 0.5
            except BadRequestError:
                pass
            self.logprobs_ok = False
        text = self.chat(messages, max_tokens=4).lower()
        return 1.0 if text.startswith("yes") else 0.0 if text.startswith("no") else 0.5


def _strip(text):
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()


def _first_json(text):
    m = re.search(r"\{.*\}", text, flags=re.S)
    return m.group(0) if m else text
