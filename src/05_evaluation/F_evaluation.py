from anthropic import Anthropic
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

JUDGES = {
    "sonnet": {
        "provider": "anthropic",
        "model": "claude-sonnet-5",
        "key_env": "ANTHROPIC_API_KEY",
        "extra": {"thinking": {"type": "adaptive"}, "output_config": {"effort": "low"}},
    },
    "luna": {
        "provider": "openai",
        "model": "gpt-6-luna",
        "key_env": "OPENAI_API_KEY",
        "extra": {"reasoning_effort": "medium"},
    },
    "sol": {
        "provider": "openai",
        "model": "gpt-6-sol",
        "key_env": "OPENAI_API_KEY",
        "extra": {"reasoning_effort": "medium"},
    },
    "grok": {
        "provider": "openai",
        "model": "grok-4.3",
        "key_env": "XAI_API_KEY",
        "base_url": "https://api.x.ai/v1",
        "extra": {"reasoning_effort": "medium"},
    },
}

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

SYSTEM_PROMPT = """You are a professional {persona} acting as an expert judge for computational creativity.
Your task is to evaluate an original satirical text written in European Portuguese, produced from a given premise and target format. The target format is a comedy sketch script in dialogue, a newspaper opinion column in prose, or a late-night TV monologue, and the text should respect it.

Think through the text's merits and flaws before assigning the scores. Then write a concise summary of that analysis (3 to 5 sentences) in European Portuguese in the 'reasoning' field.

Score each dimension with an integer from 0 to 4, following these definitions:

1. NOVELTY: the deviation from standard probabilistic patterns and cliché.
   0 = something anyone could think of; 4 = a unique perspective that others would not think of.

2. CLARITY: how easy the response is to understand.
   0 = incomprehensible; 4 = its meaning is immediately clear.

3. RELEVANCE: the connection between the topic and the response.
   0 = completely unrelated to the topic; 4 = very closely related to the topic.

4. INTELLIGENCE: the intellectual quality of the response.
   0 = lacks intellectual elements (trivial); 4 = conveys high intellect or sophistication.

5. EMPATHY: how relatable the response is.
   0 = hard to imagine the situation or the feelings (not relatable); 4 = the situation and feelings are well understood (highly relatable).

6. CULTURAL RESONANCE (Portuguese context): the correct usage of Portuguese idioms, cultural references (e.g. football, local politics) and social register. Penalise translations of generic humour that do not fit the Portuguese context.
   The text must be written in European Portuguese, and Brazilian Portuguese is a serious fault. Look for:
   - "você" as a generic or impersonal "you" (European Portuguese uses "tu", "vocês", the impersonal "se" or "uma pessoa"); "você" as formal address to a specific person is acceptable
   - "a gente" as the default first person plural instead of "nós"
   - the gerund progressive ("estou fazendo") instead of "estou a fazer"
   - Brazilian vocabulary such as "ônibus", "celular", "trem", "banheiro", "geladeira", "café da manhã", "gol", "legal" (meaning "cool"), "pra"
   - Brazilian grammar such as "em uma" instead of "numa", a possessive without its article ("minha casa" instead of "a minha casa"), or a clitic pronoun opening a sentence ("Me diz")
   0 = the response does not feel adapted to the Portuguese context, or it reads as Brazilian Portuguese; 4 = it could have been written by a native of Portugal.

7. OVERALL SCORE: integrates all the dimensions above into a single judgement, including cultural resonance, rather than measuring humour alone.
   0 = not funny in the Portuguese context; 4 = extremely funny in the Portuguese context.

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


class Judge:
    def __init__(self, name: str):
        if name not in JUDGES:
            raise ValueError(f"Unknown judge '{name}'. Choose from {list(JUDGES)}.")
        self.name = name
        self.cfg = JUDGES[name]

        api_key = os.getenv(self.cfg["key_env"])
        if not api_key:
            raise ValueError(f"Missing {self.cfg['key_env']} in .env file.")

        if self.cfg["provider"] == "anthropic":
            self.client = Anthropic(api_key=api_key)
        else:
            self.client = OpenAI(api_key=api_key, base_url=self.cfg.get("base_url"))

    def complete(self, system: str, user: str, max_tokens: int, json_mode: bool = False) -> str:
        temperature = self.cfg.get("temperature")

        if self.cfg["provider"] == "anthropic":
            kwargs = {
                "model": self.cfg["model"],
                "system": system,
                "messages": [{"role": "user", "content": user}],
                "max_tokens": max_tokens,
            }
            if temperature is not None:
                kwargs["temperature"] = temperature
            kwargs.update(self.cfg.get("extra", {}))
            response = self.client.messages.create(**kwargs)
            return "".join(block.text for block in response.content if block.type == "text")

        kwargs = {
            "model": self.cfg["model"],
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "max_completion_tokens": max_tokens,
            **self.cfg.get("extra", {}),
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""


def parse_scores(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in response")

    data = json.loads(text[start : end + 1])
    result = {"reasoning": str(data.get("reasoning", ""))}
    for m in METRICS:
        value = float(data[m])
        if not 0 <= value <= 4:
            raise ValueError(f"{m}={value} outside 0-4")
        result[m] = value
    return result


class ComedyScriptEvaluator:
    def __init__(
        self,
        judge: Judge,
        input_dir="../../data/06_comedia_outputs",
        output_dir="../../data/07_eval_results",
        repeats=3,
        limit=None,
        retries=3,
    ):
        script_dir = Path(__file__).resolve().parent
        self.judge = judge
        self.repeats = repeats
        self.limit = limit
        self.retries = retries
        self.input_dir = (script_dir / input_dir).resolve()
        self.output_dir = (script_dir / output_dir / judge.name).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Judge: {judge.name} ({judge.cfg['model']}) | results in {self.output_dir}")

    def _with_retries(self, fn):
        for attempt in range(1, self.retries + 1):
            try:
                return fn()
            except Exception as e:
                if attempt == self.retries:
                    raise
                wait = 2**attempt
                logger.warning(f"Attempt {attempt} failed ({e}); retrying in {wait}s.")
                time.sleep(wait)

    def evaluate_with_persona(self, persona: str, format_type: str, premise: str, script: str):
        user_content = f"--- TARGET FORMAT ---\n{format_type}\n\n--- TARGET PREMISE ---\n{premise}\n\n--- GENERATED TEXT ---\n{script}"
        system = SYSTEM_PROMPT.format(persona=persona)
        try:
            raw = self._with_retries(lambda: self.judge.complete(system, user_content, 8000, json_mode=True))
            return parse_scores(raw)
        except Exception as e:
            logger.error(f"Judgement failed ({persona}): {e}")
            return None

    def summarize_architecture(self, arch_name: str, reasonings: list, summary_type: str = "flaws") -> str:
        logger.info(f"Generating {summary_type} summary for: {arch_name}...")
        combined_text = "\n\n".join(reasonings)
        system = "És um analista académico de IA a avaliar modelos de geração de humor e sátira."

        if summary_type == "flaws":
            focus = (
                "Foca-te EXCLUSIVAMENTE nos defeitos, limitações e falhas apontadas (ex: falta de ressonância "
                "cultural, loops de repetição, humor cliché, problemas de estrutura).\n"
                "Escreve um resumo conciso e profissional em Português de Portugal (máximo 4 frases) que "
                "diagnostique as principais fraquezas desta arquitetura."
            )
        else:
            focus = (
                "Foca-te EXCLUSIVAMENTE nas qualidades, pontos fortes e acertos detetados (ex: excelente fluidez "
                "de diálogo, forte ressonância cultural portuguesa, criatividade nas premissas).\n"
                "Escreve um resumo conciso e profissional em Português de Portugal (máximo 4 frases) que destaque "
                "os principais trunfos desta arquitetura."
            )

        user = (
            f"Analisa as seguintes avaliações feitas por vários júris sobre a arquitetura '{arch_name}'. "
            "Cada avaliação vem identificada com o tema do texto, o júri e a pontuação global (0 a 4).\n"
            f"{focus}\n\nAVALIAÇÕES:\n{combined_text}"
        )
        try:
            return self._with_retries(lambda: self.judge.complete(system, user, 4000)).strip()
        except Exception as e:
            logger.error(f"{summary_type} summary failed: {e}")
            return "Erro ao gerar o resumo."

    def run_evaluation(self):
        json_files = sorted(self.input_dir.glob("*.json"))
        if not json_files:
            logger.error(f"No generation files found in {self.input_dir}")
            return

        master_aggregated_data = []
        diagnostics_path = self.output_dir / "Diagnostics.md"
        diagnostics_path.write_text(f"# Diagnostic summaries ({self.judge.cfg['model']})\n\n", encoding="utf-8")

        for file_path in json_files:
            arch_name = file_path.stem.replace("_outputs", "")
            logger.info(f"=== Starting Evaluation for Architecture: {arch_name} ===")

            with open(file_path, "r", encoding="utf-8") as f:
                generations = json.load(f)
            if self.limit:
                generations = generations[: self.limit]

            arch_persona_records = []
            all_reasonings = []
            raw_path = self.output_dir / f"{arch_name}_raw_judgements.jsonl"

            with raw_path.open("w", encoding="utf-8") as raw_f:
                for item in tqdm(generations, desc=f"Evaluating {arch_name}", unit="item"):
                    premise = item.get("prompt", "")
                    script = item.get("output") or item.get("edited_output", "")
                    theme = item.get("theme", "Unknown")
                    format_type = item.get("format", "sketch")

                    if not script or script.startswith("ERROR"):
                        continue

                    for persona in PERSONAS:
                        runs = []
                        for rep in range(self.repeats):
                            result = self.evaluate_with_persona(persona, format_type, premise, script)
                            raw_f.write(
                                json.dumps(
                                    {
                                        "architecture": arch_name,
                                        "judge": self.judge.cfg["model"],
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
                            if result:
                                runs.append(result)

                        if not runs:
                            logger.warning(f"{persona}: no valid judgements for '{theme}'.")
                            continue

                        reasoning_text = runs[0]["reasoning"]

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
                            values = [r[m] for r in runs]
                            record[m] = sum(values) / len(values)
                            record[f"{m}_rep_sd"] = pd.Series(values).std(ddof=1) if len(values) > 1 else 0.0

                        arch_persona_records.append(record)
                        all_reasonings.append(
                            f"[{theme} | {persona} | global {record['overall_score']:.1f}] {reasoning_text}"
                        )

            if not arch_persona_records:
                logger.warning(f"{arch_name}: no valid evaluations; skipping.")
                continue

            df_arch = pd.DataFrame(arch_persona_records)
            df_arch.to_csv(self.output_dir / f"{arch_name}_individual_personas.csv", index=False, encoding="utf-8")

            per_item = df_arch.groupby("Theme")[METRICS].mean()
            per_item.round(2).to_csv(self.output_dir / f"{arch_name}_per_item.csv", encoding="utf-8")

            avg_record = {
                "Architecture": arch_name,
                "Judge": self.judge.cfg["model"],
                "N_Items": len(per_item),
                "N_Evaluations": len(df_arch),
                "N_Repeats": self.repeats,
            }
            for m in METRICS:
                avg_record[m] = round(per_item[m].mean(), 2)
                avg_record[f"{m}_sd"] = round(per_item[m].std(ddof=1), 2)

            avg_record["Persona_Disagreement"] = round(df_arch.groupby("Theme")["overall_score"].std(ddof=1).mean(), 2)
            avg_record["Judge_Repeat_SD"] = round(df_arch[[f"{m}_rep_sd" for m in METRICS]].mean().mean(), 3)
            avg_record["Mean_Word_Count"] = int(df_arch["Word_count"].mean())

            strengths = self.summarize_architecture(arch_name, all_reasonings, summary_type="strengths")
            flaws = self.summarize_architecture(arch_name, all_reasonings, summary_type="flaws")
            with diagnostics_path.open("a", encoding="utf-8") as md:
                md.write(f"## {arch_name}\n\n**Pontos fortes**\n\n{strengths}\n\n**Fraquezas**\n\n{flaws}\n\n")

            master_aggregated_data.append(avg_record)

        if master_aggregated_data:
            df_master = pd.DataFrame(master_aggregated_data).sort_values(by="overall_score", ascending=False)
            master_csv_path = self.output_dir / "Master_Aggregated_Results.csv"
            df_master.to_csv(master_csv_path, index=False, encoding="utf-8")
            logger.info(f"=== Evaluation complete. Master aggregation saved to: {master_csv_path} ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComedIA LLM-as-a-Judge evaluation")
    parser.add_argument("--judge", required=True, choices=list(JUDGES), help="Judge model to use")
    parser.add_argument("--repeats", type=int, default=3, help="Judgements per persona per item")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate only the first N items per architecture")
    args = parser.parse_args()

    evaluator = ComedyScriptEvaluator(Judge(args.judge), repeats=args.repeats, limit=args.limit)
    evaluator.run_evaluation()
