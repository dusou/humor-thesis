import argparse
from datasets import load_dataset
import gc
import json
import logging
import os
from pathlib import Path
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
import re
import sys
import torch
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
    StoppingCriteria,
    StoppingCriteriaList,
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


class PresencePenaltyLogitsProcessor(LogitsProcessor):
    """
    Presence penalty: a flat subtraction from the logit of
    any token already generated. Unlike repetition_penalty this only penalises the model's
    own output (recommended by Qwen3.5 card for endless repetition)
    """

    def __init__(self, penalty: float, prompt_len: int):
        self.penalty = penalty
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores):
        generated = input_ids[:, self.prompt_len :]
        for i in range(scores.shape[0]):
            if generated[i].numel():
                scores[i, torch.unique(generated[i])] -= self.penalty
        return scores


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

    def _format_chatml(self, example: Dict[str, List[str]]) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
                {"role": "assistant", "content": example["output"]},
            ]
        }

    def train(self) -> None:
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found at {self.dataset_path}")
            return

        logger.info(f"Loading dataset from {self.dataset_path}")
        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        full_dataset = dataset.map(
            self._format_chatml,
            batched=False,
            remove_columns=dataset.column_names,
        )

        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
        logger.info(f"Training on {len(train_data)} examples, evaluating on {len(eval_data)} examples.")

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
            target_modules=[
                "down_proj",
                "gate_proj",
                "in_proj_a",
                "in_proj_b",
                "in_proj_qkv",
                "in_proj_z",
                "k_proj",
                "o_proj",
                "out_proj",
                "q_proj",
                "up_proj",
                "v_proj",
            ],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
        )

        training_args = SFTConfig(
            output_dir=str(self.output_dir / f"checkpoints_r{self.rank}"),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            learning_rate=2e-4,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            num_train_epochs=15,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=20,
            per_device_eval_batch_size=2,
            eval_accumulation_steps=1,
            save_steps=40,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            optim="adamw_torch",
            bf16=True,
            seed=42,
            warmup_ratio=0.1,
            report_to="tensorboard",
            max_length=4096,
            assistant_only_loss=True,
            dataloader_num_workers=4,
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            eval_dataset=eval_data,
            peft_config=lora_config,
            processing_class=tokenizer,
            args=training_args,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        logger.info("Starting LoRA Fine-Tuning...")
        train_result = trainer.train()

        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        logger.info("Running final evaluation on the best model checkpoint...")
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

        final_save_path = self.output_dir / self.adapter_name
        trainer.model.save_pretrained(str(final_save_path))
        tokenizer.save_pretrained(str(final_save_path))
        trainer.save_state()
        logger.info(f"Training complete. Adapter and metrics saved to {final_save_path}")


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
        self, messages: list, reasoning_budget: int = 2000, answer_budget: int = 2000
    ) -> Tuple[str, str]:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                prompt_len = inputs["input_ids"].shape[1]
                out_reasoning = self.model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    logits_processor=LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, prompt_len)]),
                    stopping_criteria=StoppingCriteriaList([ThinkCloseStoppingCriteria(self.tokenizer, prompt_len)]),
                )
            reasoning_text = self.tokenizer.decode(
                out_reasoning[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
            )
        finally:
            del inputs
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if "</think>" in reasoning_text:
            reasoning_text = reasoning_text.split("</think>")[0] + "</think>\n\n"
        else:
            logger.info("Reasoning hit its token budget. Forcing closure.")
            reasoning_text = f"<think>{reasoning_text.split('<think>')[-1]}\n</think>\n\n"

        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                prompt_len = inputs_answer["input_ids"].shape[1]
                out_answer = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=answer_budget,
                    min_new_tokens=200,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=20,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    logits_processor=LogitsProcessorList([PresencePenaltyLogitsProcessor(1.5, prompt_len)]),
                )
                answer_text = self.tokenizer.decode(
                    out_answer[0, inputs_answer["input_ids"].shape[1] :], skip_special_tokens=True
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

    def _cleanup_pass(self, format_type: str, raw_text: str) -> str:
        cleanup_prompt = f"""
                    Abaixo está um rascunho de um texto de comédia portuguesa (formato: {format_type})
                    gerado automaticamente, que pode conter alguns tipos de problemas como:
                    1. Repetição de falas ou frases no final do texto (ficou "preso" a repetir a mesma linha).
                    2. Pequenos erros de formatação, como tags de personagem malformadas (ex: "[SPEAKER_00>" em vez de "[SPEAKER_00]").
                    3. Personagens não existentes no sketch podem surgir subitamente entre parentesis retos.
                    4. Palavras que podem aparecer em Português do Brasil em vez de Português Europeu.
    
                    --- RASCUNHO ---
                    {raw_text}
                    --- FIM DO RASCUNHO ---
    
                    Tarefa: devolve o texto corrigido, removendo quaisquer repetições do final e corrigindo
                    erros de formatação. Se o texto tiver sido cortado a meio de uma repetição, termina-o de
                    forma muito breve e natural (no máximo 2 a 3 falas adicionais).
    
                    IMPORTANTE:
                    - Não alteres o conteúdo, o enredo ou o estilo do resto do texto a não ser que seja necessário para resolver os problemas acima.
                    - Não acrescentes novas personagens ou temas.
                    - Devolve apenas o texto corrigido, sem comentários, notas ou explicações.
                    - Usa Português Europeu.
                    """

        messages = [
            {
                "role": "system",
                "content": "És um editor de guiões de comédia portuguesa. A tua única tarefa é corrigir repetições e erros de formatação, mantendo tudo o resto inalterado.",
            },
            {"role": "user", "content": cleanup_prompt},
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        input_len = inputs["input_ids"].shape[1]
        cleanup_budget = min(input_len + 50, 3500)

        try:
            with self.model.disable_adapter():
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=cleanup_budget,
                        do_sample=True,
                        temperature=0.3,
                        repetition_penalty=1.05,
                        top_p=0.9,
                    )
                cleaned = self.tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
        finally:
            del inputs
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return re.sub(r"^(?:assistant\s*)+", "", cleaned.strip(), flags=re.IGNORECASE)

    def generate(self, query: str, format_type: str = "sketch") -> Dict:
        if self.dry_run or not self.model:
            return {"text": f"[DRY RUN] Generated mock {format_type} output.", "sources": []}

        try:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{MACRO_INSTRUCTION}\n\n{query}"},
            ]
            reasoning, response = self._generate_bounded(messages)
            clean_response = self._cleanup_pass(format_type, response)
            print("===/ AFTER CLEANUP /====" + clean_response)
            return {"text": clean_response, "sources": [], "reasoning": reasoning, "response_without_cleanup": response}
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
        "--rank",
        type=int,
        default=16,
        choices=[4, 8, 16, 32, 64],
        help="LoRA rank dimension. Alpha will be set to rank value.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent

    if args.mode == "train":
        logger.info(f"Initializing LoRA Trainer (Rank {args.rank})...")
        trainer = ComediaLoRATrainer(rank=args.rank)
        trainer.train()

    elif args.mode == "generate":
        set_seed(42)
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
                    "uncleaned_output": generation_result.get("response_without_cleanup", ""),
                    "output": generation_result["text"],
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Generation complete! Results saved to {output_path.name}")
