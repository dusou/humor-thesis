import argparse
import gc
import json
import logging
import os
from pathlib import Path
import re
import torch
import transformers
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

    def _generate_bounded(
        self, messages: list, reasoning_budget: int = 3000, answer_budget: int = 3000
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
                out_reasoning = self.model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    no_repeat_ngram_size=32,
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
            logger.info("Reasoning hit its token budget. Forcing closure.")
            reasoning_text = f"<think>{reasoning_text.split('<think>')[-1]}\n</think>\n\n"

        # Phase 2: Final answer generation
        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)
        try:
            with torch.inference_mode():
                out_answer = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=answer_budget,
                    do_sample=True,
                    temperature=0.7,
                    repetition_penalty=1.1,
                    top_p=0.9,
                    no_repeat_ngram_size=32,
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
        cleanup_budget = min(input_len + 100, 4600)

        try:
            with self.model.disable_adapter():
                with torch.inference_mode():
                    out = self.model.generate(
                        **inputs,
                        max_new_tokens=cleanup_budget,
                        do_sample=True,
                        temperature=0.3,
                        repetition_penalty=1.1,
                        top_p=0.9,
                        no_repeat_ngram_size=32,
                    )
                cleaned = self.tokenizer.decode(out[0, input_len:], skip_special_tokens=True)
        finally:
            del inputs, out
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        return re.sub(r"^(?:assistant\s*)+", "", cleaned.strip(), flags=re.IGNORECASE)

    def generate(self, query: str, format_type: str = "sketch") -> Dict:
        """
        Generates comedic text based on the provided query and format.
        """
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
            reasoning, response = self._generate_bounded(messages)
            clean_response = response
            clean_response = self._cleanup_pass(format_type, response)
            print("===/ AFTER CLEANUP /====" + clean_response)
            return {"text": clean_response, "sources": [], "reasoning": reasoning, "response_without_cleanup": response}

        except Exception as e:
            logger.error(f"Baseline generation failed for prompt: {query[:30]}... Reason: {e}")
            return {"text": "ERROR: Generation failed.", "sources": []}


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
                "reasoning": generation_result["reasoning"],
                "output": generation_result["text"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Baseline generation complete! Saved {len(output_data)} generations to: {output_path}")
