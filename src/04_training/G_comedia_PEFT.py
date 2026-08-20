import argparse
from datasets import load_dataset
import gc
import json
import logging
import os
from pathlib import Path
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
import re
import torch
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTConfig, SFTTrainer
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


class ComediaLoRATrainer:
    """Handles Parameter-Efficient Fine-Tuning (PEFT) of the base model."""

    def __init__(
        self,
        rank: int,
        model_name: str = "Qwen/Qwen3.5-9B",
    ) -> None:
        self.rank = rank
        self.model_name = model_name
        self.hf_token = os.getenv("HF_TOKEN")

        script_dir = Path(__file__).resolve().parent
        self.dataset_path = (script_dir / "../../data/04_lora_ready/lora_instruction_dataset.jsonl").resolve()

        # Local adapters directory
        self.output_dir = script_dir / "adapters"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_name = f"comedia_lora_r{self.rank}"

    def _format_chatml(self, example: Dict[str, List[str]]) -> List[str]:
        sys_msg = "És um argumentista profissional de comédia e sátira portuguesa."
        user_msg = f"{example['instruction']}\n\n{example['input']}"
        assistant_msg = example["output"]
        return (
            f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )

    def train(self) -> None:
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found at {self.dataset_path}")
            return

        logger.info(f"Loading dataset from {self.dataset_path}")
        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")

        logger.info(f"Initializing tokenizer for {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        logger.info("Loading base model in 4-bit precision...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.hf_token,
            attn_implementation="sdpa",
        )

        model = prepare_model_for_kbit_training(model)
        model.config.use_cache = False

        # Dynamic alpha scaling based on rank
        lora_alpha = self.rank * 2
        logger.info(f"Applying LoRA config (rank={self.rank}, alpha={lora_alpha})")

        lora_config = LoraConfig(
            r=self.rank,
            lora_alpha=lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        training_args = TrainingArguments(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            max_steps=500,
            logging_steps=10,  # Keeps standard HF ETA logs updated frequently
            save_steps=100,
            optim="paged_adamw_32bit",
            bf16=True,
            warmup_ratio=0.03,
            report_to="none",
        )

        training_args = SFTConfig(
            output_dir=str(self.output_dir / "checkpoints"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            max_steps=500,
            logging_steps=10,
            save_steps=100,
            optim="paged_adamw_32bit",
            bf16=True,
            warmup_ratio=0.03,
            report_to="none",
            max_length=2048,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=lora_config,
            processing_class=tokenizer,
            args=training_args,
            formatting_func=self._format_chatml,
        )

        logger.info("Starting LoRA Fine-Tuning...")
        trainer.train()

        final_save_path = self.output_dir / self.adapter_name
        trainer.model.save_pretrained(str(final_save_path))
        tokenizer.save_pretrained(str(final_save_path))
        logger.info(f"Training complete. Adapter saved to {final_save_path}")


class ComediaLoRAGenerator:
    """Handles inference combining the base model with the trained LoRA adapter."""

    def __init__(
        self,
        adapter_path: Path,
        base_model_name: str = "Qwen/Qwen3.5-9B",
        dry_run: bool = False,
    ) -> None:
        self.dry_run = dry_run
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.hf_token = os.getenv("HF_TOKEN")
        self.tokenizer = None
        self.model = None

        self._initialize_model(base_model_name, adapter_path)

    def _initialize_model(self, base_model_name: str, adapter_path: Path) -> None:
        if self.dry_run:
            logger.info("DRY RUN: Skipping LLM initialization.")
            return

        if not adapter_path.exists():
            logger.error(f"Adapter not found at {adapter_path}. Please train it first.")
            return

        logger.info(f"Loading Base LLM: {base_model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=self.hf_token)

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=self.compute_type,
                device_map="auto",
                token=self.hf_token,
                attn_implementation="sdpa",
            )

            logger.info(f"Merging LoRA Adapter: {adapter_path.name}")
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.eval()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.model = None

    def _generate_bounded(
        self, messages: list, reasoning_budget: int = 3000, answer_budget: int = 3000
    ) -> Tuple[str, str]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                out_reasoning = self.model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    top_p=0.9,
                )
            reasoning_text = self.tokenizer.decode(
                out_reasoning[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
            )
        finally:
            del inputs, out_reasoning
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if "</think>" in reasoning_text:
            reasoning_text = reasoning_text.split("</think>")[0] + "</think>\n\n"
        else:
            reasoning_text = f"<think>{reasoning_text.split('<think>')[-1]}\n</think>\n\n"

        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                out_answer = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=answer_budget,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    top_p=0.9,
                )
            answer_text = self.tokenizer.decode(
                out_answer[0, inputs_answer["input_ids"].shape[1] :], skip_special_tokens=True
            )
        finally:
            del inputs_answer, out_answer
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        clean_answer = re.sub(r"<think>[\s\S]*?</think>", "", answer_text)
        clean_answer = re.sub(r"</?think>", "", clean_answer)
        clean_answer = re.sub(r"^(?:assistant\s*)+", "", clean_answer.strip(), flags=re.IGNORECASE)

        return reasoning_text, clean_answer.strip()

    def generate(self, query: str, format_type: str = "sketch") -> Dict:
        format_mapping = {
            "sketch": {
                "system_instruction": "És um argumentista profissional de comédia e sátira portuguesa.",
                "task_instruction": "escreve um novo sketch de comédia para um vídeo entre 2 e 5 minutos sobre o seguinte tema",
            },
            "newspaper": {
                "system_instruction": "És um cronista satírico a escrever um artigo de opinião para um jornal português.",
                "task_instruction": "escreve um texto de opinião humorístico e satírico com entre 5 a 12 parágrafos sobre o seguinte tema",
            },
            "tv_show": {
                "system_instruction": "És o guionista de um programa de televisão humorístico estilo 'late-night' sobre a atualidade portuguesa.",
                "task_instruction": "escreve o guião de um monólogo televisivo de entre 2 a 5 minutos de duração que relata eventos reais de forma cómica sobre",
            },
        }

        instructions = format_mapping.get(format_type, format_mapping["sketch"])

        if self.dry_run or not self.model:
            return {"text": f"[DRY RUN] Generated mock {format_type} output.", "sources": []}

        try:
            messages = [
                {"role": "system", "content": f"{instructions['system_instruction']}\n..."},
                {"role": "user", "content": f"{instructions['task_instruction']}: {query}"},
            ]
            reasoning, clean_response = self._generate_bounded(messages)
            return {"text": clean_response, "sources": [], "reasoning": reasoning}
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {"text": "ERROR: Generation failed.", "sources": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComedIA PEFT Handler")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "generate"],
        required=True,
        help="Select 'train' to fine-tune or 'generate' to batch process prompts.",
    )
    parser.add_argument(
        "--rank", type=int, default=16, choices=[8, 16, 64], help="LoRA rank dimension. Alpha will be set to rank * 2."
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.mode == "train":
        logger.info(f"Initializing LoRA Trainer (Rank {args.rank})...")
        trainer = ComediaLoRATrainer(rank=args.rank)
        trainer.train()

    elif args.mode == "generate":
        logger.info(f"Initializing LoRA Generator (Target Rank {args.rank})...")
        input_path = (script_dir / "input_prompts.json").resolve()
        output_path = (script_dir / f"../../data/06_comedia_outputs/ComedIA_PEFT_r{args.rank}_outputs.json").resolve()
        adapter_path = (script_dir / "adapters" / f"comedia_lora_r{args.rank}").resolve()

        if not input_path.exists():
            logger.error(f"Input file '{input_path}' not found.")
            exit(1)

        with open(input_path, "r", encoding="utf-8") as f:
            prompts_list = json.load(f)

        logger.info(f"Loaded {len(prompts_list)} prompt pairs.")

        peft_system = ComediaLoRAGenerator(adapter_path=adapter_path, dry_run=args.dry_run)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        output_data = []

        for item in tqdm(prompts_list, desc=f"Generating (r={args.rank})", unit="prompt"):
            theme = item.get("theme", "general")
            format_type = item.get("format", "sketch")
            prompt = item.get("prompt", "")

            generation_result = peft_system.generate(query=prompt, format_type=format_type)

            output_data.append(
                {
                    "theme": theme,
                    "format": format_type,
                    "prompt": prompt,
                    "reasoning": generation_result.get("reasoning", ""),
                    "output": generation_result["text"],
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Generation complete! Results saved to {output_path.name}")
