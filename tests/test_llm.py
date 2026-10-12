from types import SimpleNamespace

import httpx
from openai import BadRequestError

from lastknown import LLM


def reply(text, logprobs=None):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), logprobs=logprobs)])


def bad_request():
    req = httpx.Request("POST", "http://x")
    return BadRequestError("unsupported", response=httpx.Response(400, request=req), body=None)


class FakeCompletions:
    def __init__(self, schema=True, logprobs=True):
        self.schema, self.logprobs, self.calls = schema, logprobs, []

    def create(self, **kw):
        self.calls.append(kw)
        if "response_format" in kw and not self.schema:
            raise bad_request()
        if kw.get("logprobs"):
            if not self.logprobs:
                return reply("No")
            tops = [SimpleNamespace(token="No", logprob=-0.1), SimpleNamespace(token="Yes", logprob=-2.4)]
            return reply("No", SimpleNamespace(content=[SimpleNamespace(top_logprobs=tops)]))
        if kw["max_tokens"] <= 4:
            return reply("Yes.")
        return reply('Sure: {"facts": []}')


def make(**kw):
    llm = LLM(base_url="http://x/v1", model="m", api_key="k")
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions(**kw)))
    return llm


def test_schema_fallback_parses_json_from_text():
    llm = make(schema=False)
    assert llm.chat([{"role": "user", "content": "x"}], schema={"type": "object"}) == {"facts": []}
    assert llm.schema_ok is False


def test_yes_prob_from_logprobs():
    p = make().yes_prob("Is it?")
    assert 0.05 < p < 0.15


def test_yes_prob_without_logprobs_falls_back_to_text():
    llm = make(logprobs=False)
    assert llm.yes_prob("Is it?") == 1.0
    assert llm.logprobs_ok is False


def test_no_thinking_switch_by_default():
    llm = make()
    llm.chat([{"role": "user", "content": "x"}])
    assert llm.client.chat.completions.calls[-1]["extra_body"] == {}
