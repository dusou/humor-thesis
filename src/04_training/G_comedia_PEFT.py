import argparse
from datasets import load_dataset
import gc
import json
import logging
import os
from pathlib import Path
from peft import LoraConfig, PeftModel
import re
import sys
import torch
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
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


class RepetitionControlProcessor(LogitsProcessor):
    """Presence is flat; frequency scales with count and is what breaks a
    running loop. Cap keeps common function words usable in long generations."""

    def __init__(self, presence, frequency, prompt_len, max_freq=4.0):
        self.presence, self.frequency = presence, frequency
        self.prompt_len, self.max_freq = prompt_len, max_freq

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len :]
        for i in range(scores.shape[0]):
            if not gen[i].numel():
                continue
            toks, counts = torch.unique(gen[i], return_counts=True)
            scores[i, toks] -= self.presence + torch.clamp(self.frequency * counts.to(scores.dtype), max=self.max_freq)
        return scores


class MaskThinkCollator:
    """Keeps the <think> block in the context but removes it from the loss."""

    def __init__(self, base_collator, tokenizer):
        self.base = base_collator
        self.close_ids = torch.tensor(tokenizer.encode("</think>", add_special_tokens=False), dtype=torch.long)

    def __call__(self, features):
        batch = self.base(features)
        ids, labels = batch["input_ids"], batch["labels"]
        n = self.close_ids.numel()
        close = self.close_ids.to(ids.device)

        for i in range(ids.size(0)):
            row = ids[i]
            # Search backwards for the last </think>.
            for j in range(row.size(0) - n, -1, -1):
                if torch.equal(row[j : j + n], close):
                    labels[i, : j + n] = -100
                    break

        batch["labels"] = labels
        return batch


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

    def _list_checkpoints(self, ckpt_dir: Path) -> list:
        rows = []
        for d in sorted(ckpt_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1])):
            state_file = d / "trainer_state.json"
            if not state_file.exists():
                continue
            state = json.loads(state_file.read_text(encoding="utf-8"))
            losses = [h["eval_loss"] for h in state.get("log_history", []) if "eval_loss" in h]
            rows.append(
                {
                    "path": d,
                    "step": int(d.name.split("-")[1]),
                    "epoch": round(state.get("epoch", 0.0), 2),
                    "eval_loss": round(losses[-1], 4) if losses else None,
                }
            )
        return rows

    def _format_chatml(self, example: Dict[str, List[str]]) -> dict:
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{example['instruction']}\n\n{example['input']}"},
                {"role": "assistant", "content": example["output"]},
            ]
        }

    def _load_split(self):
        dataset = load_dataset("json", data_files=str(self.dataset_path), split="train")
        full_dataset = dataset.map(
            self._format_chatml,
            batched=False,
            remove_columns=dataset.column_names,
        )
        return full_dataset.train_test_split(test_size=0.1, seed=42)

    def train(self) -> None:
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found at {self.dataset_path}")
            return

        logger.info(f"Loading dataset from {self.dataset_path}")
        split_dataset = self._load_split()
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
        logger.info(f"Training on {len(train_data)} examples, evaluating on {len(eval_data)} examples.")

        logger.info(f"Initializing tokenizer for {self.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        logger.info("Loading base model...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map="auto",
            token=self.hf_token,
            attn_implementation="sdpa",
        )

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
            use_rslora=False,
        )

        training_args = SFTConfig(
            output_dir=str(self.output_dir / f"checkpoints_r{self.rank}"),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            learning_rate=2e-4,
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            num_train_epochs=10,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=20,
            per_device_eval_batch_size=2,
            eval_accumulation_steps=1,
            save_steps=20,
            save_total_limit=6,
            save_only_model=True,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            optim="adamw_torch",
            bf16=True,
            seed=42,
            warmup_steps=20,
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
            callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
        )

        trainer.data_collator = MaskThinkCollator(trainer.data_collator, tokenizer)

        probe = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
        for i in range(2):
            keep = probe["labels"][i] != -100
            text = tokenizer.decode(probe["input_ids"][i][keep])
            logger.info(f"mask probe {i}: {int(keep.sum())} tokens in loss")
            logger.info(f"  starts: {text[:100]!r}")
            if int(keep.sum()) == 0:
                raise ValueError("Everything masked; the </think> search ran past the sequence.")
            if "<think>" in text or "ELENCO" in text:
                raise ValueError("Reasoning still in the loss; masking failed.")

        trainer.model.print_trainable_parameters()
        attached = {n.split(".lora_A")[0].split(".")[-1] for n, _ in trainer.model.named_modules() if ".lora_A" in n}
        logger.info(f"LoRA attached to: {sorted(attached)}")
        missing = set(lora_config.target_modules) - attached
        if missing:
            logger.warning(f"target_modules never matched: {sorted(missing)}")
        else:
            logger.info("all target_modules matched")

        logger.info("Starting LoRA Fine-Tuning...")
        train_result = trainer.train()

        train_metrics = train_result.metrics
        trainer.log_metrics("train", train_metrics)
        trainer.save_metrics("train", train_metrics)

        final_save_path = self.output_dir / self.adapter_name
        trainer.model.save_pretrained(str(final_save_path))
        tokenizer.save_pretrained(str(final_save_path))
        trainer.save_state()
        logger.info(f"Training complete. Adapter and metrics saved to {final_save_path}")

        # Eval phase
        logger.info("Measuring adapter contribution against the frozen base...")
        lora_metrics = trainer.evaluate(metric_key_prefix="lora")
        with trainer.model.disable_adapter():
            base_metrics = trainer.evaluate(metric_key_prefix="base")

        trainer.log_metrics("eval", lora_metrics)
        trainer.save_metrics("eval", lora_metrics)
        lora_loss = lora_metrics["lora_loss"]
        base_loss = base_metrics["base_loss"]
        result = {
            "rank": self.rank,
            "base_loss": base_loss,
            "lora_loss": lora_loss,
            "delta": base_loss - lora_loss,
            "base_ppl": float(torch.exp(torch.tensor(base_loss))),
            "lora_ppl": float(torch.exp(torch.tensor(lora_loss))),
            "n_eval": len(eval_data),
        }
        (self.output_dir / f"eval_r{self.rank}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        logger.info(f"base {base_loss:.4f} -> lora {lora_loss:.4f} | delta {base_loss - lora_loss:+.4f}")


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
        self, messages: list, reasoning_budget: int = 5500, answer_budget: int = 2500
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
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.1, 0.3, prompt_len)]),
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

        if "</think>" not in reasoning_text:
            logger.warning(f"Reasoning did not close within {reasoning_budget} tokens.")
        reasoning_text = self._normalise_reasoning(reasoning_text)

        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                prompt_len = inputs_answer["input_ids"].shape[1]
                out_answer = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=answer_budget,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=20,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.1, 0.3, prompt_len)]),
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
            return {
                "response": response,
                "sources": [],
                "reasoning": reasoning,
                "response_with_cleanup": clean_response,
            }
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "response": "ERROR: Generation failed.",
                "sources": [],
                "reasoning": "ERROR: Generation failed.",
                "response_with_cleanup": "ERROR: Generation failed.",
            }


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
        help="LoRA rank dimension. Alpha will be set to rank value * 2.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Override the adapter used. Accepts a checkpoint dir, e.g. adapters/checkpoints_r16/checkpoint-120",
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
        adapter_path = (
            Path(args.adapter_path).resolve()
            if args.adapter_path
            else (script_dir / "adapters" / f"comedia_lora_r{args.rank}").resolve()
        )
        tag = adapter_path.name if args.adapter_path else f"r{args.rank}"
        output_path = (script_dir / f"../../data/06_comedia_outputs/ComedIA_PEFT_{tag}_outputs.json").resolve()

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
                    "clean_output": generation_result.get("response_with_cleanup", ""),
                    "output": generation_result["response"],
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=4)

        logger.info(f"Generation complete! Results saved to {output_path.name}")
