import argparse
from dotenv import load_dotenv
import json
import logging
from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import re
import time
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

PERSONAS = [
    "Author/Novelist",
    "Content Writer",
    "Technical Writer",
    "Copywriter",
    "Editor",
    "Comedian",
    "Journalist",
    "Portuguese (Portugal) Specialist",
]

METRICS = ["novelty", "clarity", "relevance", "intelligence", "empathy", "cultural_resonance", "overall_score"]


class ComedyScriptEvaluator:
    def __init__(self, input_dir="../../data/06_comedia_outputs", output_dir="../../data/07_eval_results", repeats=3):
        script_dir = Path(__file__).resolve().parent
        self.repeats = repeats
        self.input_dir = (script_dir / input_dir).resolve()
        self.output_dir = (script_dir / output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.api_key = os.getenv("EVAL_API_KEY")
        self.base_url = os.getenv("EVAL_BASE_URL")
        self.model_id = os.getenv("EVAL_MODEL")

        if not all([self.api_key, self.base_url, self.model_id]):
            raise ValueError("API credentials missing in .env file.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        logger.info(f"Initialized API Evaluator targeting: {self.base_url} with model {self.model_id}")

    def evaluate_with_persona(self, persona: str, format_type: str, premise: str, script: str) -> dict:
        system_prompt = f"""You are a professional {persona} acting as an expert judge for computational creativity.
Your task is to evaluate an original satirical text written in European Portuguese based on a given premise and format (which can be a comedy sketch script, a newspaper opinion column, or a late-night TV monologue).
You must use a Chain-of-Thought approach: first analyze the text's merits and flaws inside a 'reasoning' block, then output numerical scores.
You must also output a summary of your reasoning in European Portuguese behind those numerical scores tagged as 'reasoning'.
Evaluate the output on a Likert scale (0 to 4) across these exact 7 dimensions:
1. Novelty (0 = cliché, 4 = highly creative/subversive)
2. Clarity (0 = chaotic structure, 4 = narrative flow and formatting match the intended format perfectly)
3. Relevance (0 = unrelated, 4 = perfectly addresses the premise)
4. Intelligence (0 = trivial, 4 = sophisticated satirical logic)
5. Empathy (0 = flat tone, 4 = highly relatable human perspective)
6. Cultural Resonance (0 = feels like an American translation, 4 = flawless usage of Portuguese idioms, social register, and national context)
7. Overall Score (0 = not funny/effective, 4 = extremely witty and production-ready)

CRITICAL RULE: Return ONLY a valid JSON object.
{{
    "reasoning": "Step-by-step analysis here FIRST...",
    "novelty": 0,
    "clarity": 0,
    "relevance": 0,
    "intelligence": 0,
    "empathy": 0,
    "cultural_resonance": 0,
    "overall_score": 0
}}"""
        user_content = f"--- TARGET FORMAT ---\n{format_type}\n\n--- TARGET PREMISE ---\n{premise}\n\n--- GENERATED TEXT ---\n{script}"

        try:
            time.sleep(1)
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.3,
                max_completion_tokens=2048,
                response_format={"type": "json_object"},
            )
            raw_output = response.choices[0].message.content.strip()
            clean_text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)
            if "<think>" in clean_text:
                clean_text = clean_text.split("<think>")[0]

            return json.loads(clean_text.strip())
        except Exception as e:
            logger.error(f"API Error during persona assessment ({persona}): {e}")
            return None

    def summarize_architecture(self, arch_name: str, reasonings: list, summary_type: str = "flaws") -> str:
        """Synthesizes all persona reasoning blocks into a concise Portuguese diagnostic summary."""
        logger.info(f"Generating architecture {summary_type} summary for: {arch_name}...")
        combined_text = "\n\n--- NEXT EVALUATION ---\n\n".join(reasonings)[:20000]

        system_prompt = "És um analista académico de IA a avaliar modelos de geração de humor e sátira."

        if summary_type == "flaws":
            user_prompt = f"""Analisa as seguintes avaliações feitas por vários júris sobre a arquitetura '{arch_name}'. 
Foca-te EXCLUSIVAMENTE nos defeitos, limitações e falhas apontadas (ex: falta de ressonância cultural, loops de repetição, humor cliché, problemas de estrutura).
Escreve um resumo conciso e profissional em Português de Portugal (máximo 4 frases) que diagnostique as principais fraquezas desta arquitetura.

AVALIAÇÕES:
{combined_text}
"""
        else:
            user_prompt = f"""Analisa as seguintes avaliações feitas por vários júris sobre a arquitetura '{arch_name}'. 
Foca-te EXCLUSIVAMENTE nas qualidades, pontos fortes e acertos detetados (ex: excelente fluidez de diálogo, forte ressonância cultural portuguesa, criatividade nas premissas).
Escreve um resumo conciso e profissional em Português de Portugal (máximo 4 frases) que destaque os principais trunfos desta arquitetura.

AVALIAÇÕES:
{combined_text}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.4,
                max_completion_tokens=500,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"API Error during {summary_type} summarization: {e}")
            return "Erro ao gerar o resumo."

    def run_evaluation(self):
        json_files = list(self.input_dir.glob("*.json"))
        if not json_files:
            logger.error(f"No generation files found in {self.input_dir}")
            return

        master_aggregated_data = []

        for file_path in json_files:
            arch_name = file_path.stem.replace("_outputs", "")
            raw_path = self.output_dir / f"{arch_name}_raw_judgements.jsonl"
            diagnostics_path = self.output_dir / "Diagnostics.md"
            diagnostics_path.write_text("# Diagnostic summaries\n\n", encoding="utf-8")
            with raw_path.open("w", encoding="utf-8") as raw_f:
                logger.info(f"=== Starting Evaluation for Architecture: {arch_name} ===")

                with open(file_path, "r", encoding="utf-8") as f:
                    generations = json.load(f)

                arch_persona_records = []
                all_reasonings = []

                for item in tqdm(generations, desc=f"Evaluating {arch_name}", unit="item"):
                    premise = item.get("prompt", "")
                    script = item.get("output", "")
                    theme = item.get("theme", "Unknown")
                    format_type = item.get("format", "sketch")

                    if not script or script == "ERROR: Generation failed.":
                        continue

                    for persona in PERSONAS:
                        runs = []
                        for rep in range(self.repeats):
                            result = self.evaluate_with_persona(persona, format_type, premise, script)

                            raw_f.write(
                                json.dumps(
                                    {
                                        "architecture": arch_name,
                                        "theme": theme,
                                        "format": format_type,
                                        "persona": persona,
                                        "repeat": rep,
                                        "premise": premise,
                                        "word_count": len(script.split()),
                                        "response": result,
                                    },
                                    ensure_ascii=False,
                                )
                                + "\n"
                            )
                            raw_f.flush()

                            if not result:
                                continue
                            missing = [m for m in METRICS if m not in result]
                            if missing:
                                logger.warning(f"{persona} rep {rep}: missing {missing}.")
                                continue
                            runs.append(result)

                        if not runs:
                            logger.warning(f"{persona}: no valid judgements for '{theme}'.")
                            continue

                        reasoning_text = runs[0].get("reasoning", "")
                        all_reasonings.append(reasoning_text)

                        if persona == "Portuguese (Portugal) Specialist":
                            print(
                                f"\n\033[96m--- PT Specialist Review [{format_type.upper()}] | Theme: {theme} ---\033[0m"
                            )
                            print(f"\033[93m{reasoning_text}\033[0m\n")

                        record = {
                            "Architecture": arch_name,
                            "Format": format_type,
                            "Theme": theme,
                            "Persona": persona,
                            "Reasoning": reasoning_text,
                            "Word_count": len(script.split()),
                            "N_Repeats": len(runs),
                        }

                        for m in METRICS:
                            values = [float(r[m]) for r in runs]
                            record[m] = sum(values) / len(values)
                            record[f"{m}_rep_sd"] = pd.Series(values).std(ddof=1) if len(values) > 1 else 0.0

                        arch_persona_records.append(record)

                if not arch_persona_records:
                    logger.warning(f"{arch_name}: no valid evaluations; skipping.")
                    continue

                df_arch = pd.DataFrame(arch_persona_records)
                df_arch.to_csv(
                    self.output_dir / f"{arch_name}_individual_personas.csv",
                    index=False,
                    encoding="utf-8",
                )

                per_item = df_arch.groupby("Theme")[METRICS].mean()

                avg_record = {
                    "Architecture": arch_name,
                    "N_Items": len(per_item),
                    "N_Evaluations": len(df_arch),
                    "N_Repeats": self.repeats,
                }
                for m in METRICS:
                    avg_record[m] = round(per_item[m].mean(), 2)
                    avg_record[f"{m}_sd"] = round(per_item[m].std(ddof=1), 2)

                avg_record["Persona_Disagreement"] = round(
                    df_arch.groupby("Theme")["overall_score"].std(ddof=1).mean(), 2
                )
                avg_record["Mean_Word_Count"] = int(df_arch["Word_count"].mean())

                strengths = self.summarize_architecture(arch_name, all_reasonings, summary_type="strengths")
                flaws = self.summarize_architecture(arch_name, all_reasonings, summary_type="flaws")

                with (self.output_dir / "Diagnostics.md").open("a", encoding="utf-8") as md:
                    md.write(
                        f"## {arch_name}\n\n**Pontos fortes**\n\n{strengths}\n\n**Fraquezas**\n\n{flaws}\n\n---\n\n"
                    )

                avg_record["Judge_Repeat_SD"] = round(df_arch[[f"{m}_rep_sd" for m in METRICS]].mean().mean(), 3)
                master_aggregated_data.append(avg_record)

                per_item.round(2).to_csv(self.output_dir / f"{arch_name}_per_item.csv", encoding="utf-8")

        if master_aggregated_data:
            df_master = pd.DataFrame(master_aggregated_data)
            master_csv_path = self.output_dir / "Master_Aggregated_Results.csv"

            if "overall_score" in df_master.columns:
                df_master = df_master.sort_values(by="overall_score", ascending=False)

            df_master.to_csv(master_csv_path, index=False, encoding="utf-8")
            logger.info(f"=== Evaluation Complete. Master aggregation saved to: {master_csv_path.name} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComedIA LLM-as-a-Judge evaluation")
    parser.add_argument("--repeats", type=int, default=3, help="Judgements per persona per item")
    args = parser.parse_args()

    evaluator = ComedyScriptEvaluator(repeats=args.repeats)
    evaluator.run_evaluation()
