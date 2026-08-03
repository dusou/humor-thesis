from dotenv import load_dotenv
import gc
import json
import logging
import os
import shutil
import torch
from tqdm import tqdm
import transformers

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()

CANDIDATE_MODELS = {
    # "Amalia-9B": "amalia-llm/AMALIA-9B-0626-DPO",
    # "Amalia-9B-SFT": "amalia-llm/AMALIA-9B-0626-SFT",
    # "Ministral-3-8B": "mistralai/Ministral-3-8B-Reasoning-2512",
    # "Llama-3.1-8B-Instruct": "meta-llama/Llama-3.1-8B-Instruct",
    # "Llama-3.2-3B-Instruct": "meta-llama/Llama-3.2-3B-Instruct",
    # "Qwen-3.5-9B": "Qwen/Qwen3.5-9B",
    # "Gemma-4-E4B": "google/gemma-4-E4B-it",
    # "DeepSeek-R1-Distill-Llama-8B": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    # "gervasio-8b-portuguese-ptpt-decoder": "PORTULAN/gervasio-8b-portuguese-ptpt-decoder",
    # "Phi-4-mini-reasoning": "microsoft/Phi-4-mini-reasoning",
    # "Phi-4-mini-instruct": "microsoft/Phi-4-mini-instruct"
}

# Localized Portuguese comedy premises to evaluate true standalone usability
PREMISES = [
    {
        "id": "bureaucracy",
        "topic": "Uma fila de espera interminável numa repartição pública das Finanças onde o funcionário é excessivamente zeloso com carimbos.",
    },
    {
        "id": "football",
        "topic": "Dois adeptos de futebol rivais que são forçados a assistir a um derby importante juntos porque ficaram presos num elevador.",
    },
    {
        "id": "politics",
        "topic": "Um debate autárquico numa pequena vila do interior de Portugal onde os candidatos prometem coisas absurdas para ganhar votos.",
    },
    {
        "id": "family",
        "topic": "Um almoço de domingo em família onde a avó tenta explicar ao neto como funciona o 'TikTok' usando metáforas da agricultura.",
    },
]


class StandaloneScriptEvaluator:
    def __init__(self, output_dir="data/04_eval_generations"):
        script_dir = os.path.dirname(os.path.abspath(__file__))

        if output_dir is None:
            self.output_dir = os.path.normpath(
                os.path.join(script_dir, "../../data/04_eval_generations")
            )
        else:
            self.output_dir = output_dir

        logging.info(f"Targeting output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)

        load_dotenv()

    def run_generation(self):
        results = {p["topic"]: {"id": p["id"]} for p in PREMISES}

        for model_name, model_id in CANDIDATE_MODELS.items():
            logger.info(f"--- Loading {model_name} ({model_id}) ---")

            temp_cache_dir = os.path.normpath(
                os.path.join(self.output_dir, f"temp_cache_{model_name}")
            )
            os.makedirs(temp_cache_dir, exist_ok=True)

            os.environ["HF_HOME"] = temp_cache_dir

            pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                device_map="auto",
                torch_dtype=torch.float16,
            )

            for case in tqdm(PREMISES, desc=f"Generating sketches with {model_name}"):
                prompt = f"""Escreve um sketch de comédia original em Português de Portugal com base na seguinte premissa:
CENA: {case["topic"]}

Instruções:
- Cria um diálogo dinâmico e humorístico entre as personagens.
- Inclui indicações cénicas breves (didascálias).
- Mantém o tom focado na sátira social portuguesa.
- O diálogo deve ser longo o suficiente para servir como guião para um vídeo de entre 3 a 5 minutos.
"""

                prefill_text = "### Título do Sketch:"

                messages = [
                    {
                        "role": "system",
                        "content": "És um argumentista profissional de comédia satírica em Português de Portugal.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": prefill_text},
                ]

                try:
                    outputs = pipeline(
                        messages,
                        max_new_tokens=1000,
                        temperature=0.8,
                        do_sample=True,
                        continue_final_message=True,
                    )

                    full_generation = outputs[0]["generated_text"][-1]["content"].strip()
                    print(full_generation)

                    if not full_generation.startswith("### Título"):
                        full_generation = f"{prefill_text} {full_generation}"

                    results[case["topic"]][model_name] = full_generation

                except Exception as e:
                    logger.error(f"Generation failed for {model_name}: {e}")
                    results[case["topic"]][model_name] = "ERROR"

            del pipeline
            gc.collect()
            torch.cuda.empty_cache()

            logger.info(f"Freeing disk space: Deleting {temp_cache_dir}...")
            shutil.rmtree(temp_cache_dir, ignore_errors=True)

            output_path = f"{self.output_dir}/{model_name}_eval_scripts.json"
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=4)
            logger.info("Saved scripts.")
        logger.info(f"Evaluation dataset compiled successfully at: {output_path}")


if __name__ == "__main__":
    evaluator = StandaloneScriptEvaluator()
    evaluator.run_generation()
