import re
import torch
from transformers import LogitsProcessor, StoppingCriteria

REASONING_BUDGET = 5000
ANSWER_BUDGET = 2250

REASONING_SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 50,
}

ANSWER_SAMPLING = {
    "do_sample": True,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
}

PRESENCE_PENALTY = 1.05
FREQUENCY_PENALTY = 0.3


class ThinkCloseStoppingCriteria(StoppingCriteria):
    """
    Thinking Block stopping criteria
    """

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        tail_ids = input_ids[0, self.prompt_len :]
        tail_text = self.tokenizer.decode(tail_ids[-8:], skip_special_tokens=False)
        return "</think>" in tail_text


class RepetitionControlProcessor(LogitsProcessor):
    """Penalise repetition within a recent window only."""

    def __init__(
        self,
        prompt_len,
        presence=PRESENCE_PENALTY,
        frequency=FREQUENCY_PENALTY,
        max_freq=4.0,
        window=64,
    ):
        self.presence, self.frequency = presence, frequency
        self.prompt_len, self.max_freq, self.window = prompt_len, max_freq, window

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len :]
        for i in range(scores.shape[0]):
            recent = gen[i][-self.window :]
            if not recent.numel():
                continue
            toks, counts = torch.unique(recent, return_counts=True)
            scores[i, toks] -= self.presence + torch.clamp(self.frequency * counts.to(scores.dtype), max=self.max_freq)
        return scores


def normalise_reasoning(reasoning_text: str, trim_incomplete: bool = True) -> str:
    closed = "</think>" in reasoning_text

    body = reasoning_text.split("</think>")[0]
    body = re.sub(r"</?think>", "", body).strip()

    if not closed and trim_incomplete:
        cut = max(body.rfind(". "), body.rfind(".\n"), body.rfind("! "), body.rfind("? "))
        if cut > 200:
            body = body[: cut + 1]

    return f"<think>\n{body}\n</think>\n\n"


def clean_answer(answer_text: str) -> str:
    text = re.sub(r"<think>[\s\S]*?</think>", "", answer_text)
    text = re.sub(r"</?think>", "", text)
    return re.sub(r"^(?:assistant\s*)+", "", text.strip(), flags=re.IGNORECASE).strip()
