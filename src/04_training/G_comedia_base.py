import argparse
import gc
import json
import logging
import os
from pathlib import Path
import re
import sys
import torch
import transformers
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Optimize PyTorch memory allocation and suppress unnecessary telemetry
transformers.logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


data_process_dir = (Path(__file__).resolve().parent.parent / "02_data_process").resolve()
if str(data_process_dir) not in sys.path:
    sys.path.insert(0, str(data_process_dir))

from D_data_transform import MACRO_INSTRUCTION, SYSTEM_PROMPT


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

    def __init__(self, presence, frequency, prompt_len, max_freq=4.0, window=64):
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


class ComediaBaseline:
    """
    Implements the Baseline Architecture using local Qwen-3.5-9B.
    """

    def __init__(self, llm_model_name: str = "Qwen/Qwen3.5-9B", dry_run: bool = False) -> None:
        self.dry_run = dry_run
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.hf_token = os.getenv("HF_TOKEN")

        self.tokenizer = None
        self.model = None

        self._initialize_model(llm_model_name)

    def _initialize_model(self, llm_model_name: str) -> None:
        """Loads the tokenizer and generative model with memory optimizations."""
        if self.dry_run:
            logger.info("DRY RUN: Skipping LLM initialization.")
            return

        logger.info(f"Loading Generative LLM (Baseline): {llm_model_name}...")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_model_name, token=self.hf_token)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                dtype=self.compute_type,
                device_map="auto",
                token=self.hf_token,
                attn_implementation="sdpa",
            )
        except Exception as e:
            logger.error(f"Failed to load Generative LLM: {e}")
            self.model = None

    @staticmethod
    def _normalise_reasoning(reasoning_text: str, trim_incomplete: bool = True) -> str:
        closed = "</think>" in reasoning_text

        body = reasoning_text.split("</think>")[0]
        body = re.sub(r"</?think>", "", body).strip()

        if not closed and trim_incomplete:
            cut = max(body.rfind(". "), body.rfind(".\n"), body.rfind("! "), body.rfind("? "))
            if cut > 200:
                body = body[: cut + 1]

        return f"<think>\n{body}\n</think>\n\n"

    def _generate_bounded(
        self, messages: list, reasoning_budget: int = 5000, answer_budget: int = 2250
    ) -> Tuple[str, str]:
        """
        Executes a two-phase generation process: capped reasoning followed by a guaranteed answer.
        """
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        # Phase 1: Reasoning block generation
        try:
            with torch.inference_mode():
                prompt_len = inputs["input_ids"].shape[1]
                out1 = self.model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.05, 0.3, prompt_len)]),
                    stopping_criteria=StoppingCriteriaList([ThinkCloseStoppingCriteria(self.tokenizer, prompt_len)]),
                )
            reasoning_text = self.tokenizer.decode(out1[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        finally:
            del inputs
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if "</think>" not in reasoning_text:
            logger.warning(f"Reasoning did not close within {reasoning_budget} tokens.")
        reasoning_text = self._normalise_reasoning(reasoning_text)

        # Phase 2: Final answer generation
        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)
        try:
            with torch.inference_mode():
                prompt_len = inputs_answer["input_ids"].shape[1]
                out2 = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=answer_budget,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.05, 0.3, prompt_len)]),
                )
            answer_text = self.tokenizer.decode(
                out2[0, inputs_answer["input_ids"].shape[1] :], skip_special_tokens=True
            )
        finally:
            del inputs_answer
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        clean_answer = re.sub(r"<think>[\s\S]*?</think>", "", answer_text)
        clean_answer = re.sub(r"</?think>", "", clean_answer)
        clean_answer = re.sub(r"^(?:assistant\s*)+", "", clean_answer.strip(), flags=re.IGNORECASE)

        clean_reasoning = re.sub(r"</?think>", "", reasoning_text).strip()

        return clean_reasoning, clean_answer.strip()

    def generate(self, query: str, format_type: str = "sketch") -> Dict:
        if self.dry_run:
            return {"text": f"[DRY RUN] Generated mock {format_type} output.", "sources": []}

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{MACRO_INSTRUCTION}\n\n{query}"},
            ]
            reasoning, response = self._generate_bounded(messages)
            return {
                "response": response,
                "sources": [],
                "reasoning": reasoning,
            }
        except Exception as e:
            logger.error(f"Baseline generation failed: {e}")
            return {
                "response": "ERROR: Generation failed.",
                "sources": [],
                "reasoning": "ERROR: Generation failed.",
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComedIA Baseline Batch Generator")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading the Qwen model")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    input_path = os.path.normpath(script_dir / "input_prompts.json")
    output_path = os.path.normpath(script_dir / "../../data/06_comedia_outputs/ComedIA_Baseline_outputs.json")

    if not Path(input_path).exists():
        logger.error(f"Input file '{input_path}' not found.")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)

    logger.info(f"Loaded {len(prompts_list)} prompt pairs from {input_path}")

    baseline_system = ComediaBaseline(llm_model_name="Qwen/Qwen3.5-9B", dry_run=args.dry_run)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_data = []
    total_items = len(prompts_list)
    set_seed(42)

    for idx, item in enumerate(prompts_list, start=1):
        theme = item.get("theme", "general")
        format_type = item.get("format", "sketch")
        prompt = item.get("prompt", "")

        logger.info(f"Processing ({idx}/{total_items}) | Theme: '{theme}' | Format: '{format_type}'")

        generation_result = baseline_system.generate(query=prompt, format_type=format_type)

        output_data.append(
            {
                "theme": theme,
                "format": format_type,
                "prompt": prompt,
                "reasoning": generation_result.get("reasoning", ""),
                "output": generation_result["response"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Baseline generation complete! Saved {len(output_data)} generations to: {output_path}")
