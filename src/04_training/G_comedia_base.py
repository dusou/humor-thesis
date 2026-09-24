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
    LogitsProcessorList,
    set_seed,
    StoppingCriteriaList,
)
from typing import Dict, Tuple

src_dir = str(Path(__file__).resolve().parents[1])
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from common import (
    ANSWER_BUDGET,
    ANSWER_SAMPLING,
    clean_answer,
    get_instruction,
    normalise_reasoning,
    REASONING_BUDGET,
    REASONING_SAMPLING,
    RepetitionControlProcessor,
    SYSTEM_PROMPT,
    ThinkCloseStoppingCriteria,
)

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

    def _generate_bounded(self, messages: list) -> Tuple[str, str]:
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
                    max_new_tokens=REASONING_BUDGET,
                    **REASONING_SAMPLING,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(prompt_len)]),
                    stopping_criteria=StoppingCriteriaList([ThinkCloseStoppingCriteria(self.tokenizer, prompt_len)]),
                )
            reasoning_text = self.tokenizer.decode(out1[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        finally:
            del inputs
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if "</think>" not in reasoning_text:
            logger.warning(f"Reasoning did not close within {REASONING_BUDGET} tokens.")
        reasoning_text = normalise_reasoning(reasoning_text)

        # Phase 2: Final answer generation
        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)
        try:
            with torch.inference_mode():
                prompt_len = inputs_answer["input_ids"].shape[1]
                out2 = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=ANSWER_BUDGET,
                    **ANSWER_SAMPLING,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(prompt_len)]),
                )
            answer_text = self.tokenizer.decode(
                out2[0, inputs_answer["input_ids"].shape[1] :], skip_special_tokens=True
            )
        finally:
            del inputs_answer
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return re.sub(r"</?think>", "", reasoning_text).strip(), clean_answer(answer_text)

    def generate(self, query: str, format_type: str = "sketch") -> Dict:
        if self.dry_run:
            return {"text": f"[DRY RUN] Generated mock {format_type} output.", "sources": []}

        instruction = get_instruction(format_type)

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{instruction}\n\n{query}"},
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
    set_seed(42)

    baseline_system = ComediaBaseline(llm_model_name="Qwen/Qwen3.5-9B", dry_run=args.dry_run)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    output_data = []
    total_items = len(prompts_list)

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
