from dotenv import load_dotenv
import glob
import json
import logging
from openai import OpenAI
import os
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
    "Public Relations Specialist",
    "Portuguese (Portugal) Specialist",
]


class ComedyScriptEvaluator:
    def __init__(self, input_dir="data/04_eval_generations", output_dir="data/05_eval_results"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        self.input_dir = (
            os.path.normpath(os.path.join(script_dir, "../../", input_dir))
            if not os.path.isabs(input_dir)
            else input_dir
        )
        self.output_dir = (
            os.path.normpath(os.path.join(script_dir, "../../", output_dir))
            if not os.path.isabs(output_dir)
            else output_dir
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.api_key = os.getenv("EVAL_API_KEY")
        self.base_url = os.getenv("EVAL_BASE_URL")
        self.model_id = os.getenv("EVAL_MODEL")

        if not self.api_key or not self.base_url or not self.model_id:
            logger.error("Missing EVAL_API variables in environment variables.")
            raise ValueError("API credentials missing.")

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        logger.info(f"Initialized API Evaluator targeting endpoint: {self.base_url}")

    def evaluate_with_persona(self, persona, premise, script):
        """Queries the Groq API simulating a specific persona using multipart/form-data."""

        system_prompt = f"""You are a professional {
            persona
        } acting as an expert judge for computational creativity.
                        Your task is to evaluate an original comedy sketch script written in European Portuguese based on a given premise.
                        You must use a Chain-of-Thought approach: first analyze the script's merits and flaws inside a 'reasoning' block, then output numerical scores.
                        You must also output the a summary of your reasoning in European Portuguese behind those numerical scores tagged as 'reasoning'.

                        Evaluate the script on a Likert scale (0 to 4) across these exact 7 dimensions:
                        1. Novelty (0 = cliché/predictable, 4 = unique perspective/highly creative)
                        2. Clarity (0 = incomprehensible/chaotic structure, 4 = meaning and scene flow immediately clear)
                        3. Relevance (0 = completely unrelated, 4 = very closely related to the target premise)
                        4. Intelligence (0 = trivial/slapstick, 4 = high intellect/sophisticated comedic logic)
                        5. Empathy (0 = not relatable/flat characters, 4 = situation and feelings highly relatable)
                        6. Cultural Resonance (0 = feels like an American translation, 4 = flawless usage of Portuguese idioms)
                        7. Overall Score (0 = not funny in the Portuguese context, 4 = extremely funny/production-ready)

                        Return only a valid JSON object with the following keys: novelty, clarity, relevance, intelligence, empathy, cultural_resonance, overall_score, reasoning.
                        The JSON output should have 8 lowercase keys, 7 of them being the dimensions and the eighth being the reasoning summary.

                        CRITICAL RULE:Return ONLY a valid JSON object. You MUST use this exact structure, in this exact order:
                        {{
                            "reasoning": "Write your step-by-step analysis here FIRST...",
                            "novelty": "Insert value here",
                            "clarity": "Insert value here",
                            "relevance": "Insert value here",
                            "intelligence": "Insert value here",
                            "empathy": "Insert value here",
                            "cultural_resonance": "Insert value here",
                            "overall_score": "Insert value here",
                        }}
                        """

        user_content = (
            f"--- TARGET PREMISE ---\n{premise}\n\n--- GENERATED COMEDY SCRIPT ---\n{script}"
        )

        try:
            time.sleep(3)

            response = self.client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.7,
                max_completion_tokens=4096,
                top_p=1,
                reasoning_effort="medium",
                response_format={"type": "json_object"},
            )

            raw_output = response.choices[0].message.content.strip()

            clean_text = re.sub(r"<think>.*?</think>", "", raw_output, flags=re.DOTALL)

            if "<think>" in clean_text:
                clean_text = clean_text.split("<think>")[0]

            clean_json = clean_text.strip()

            return json.loads(clean_json)

        except Exception as e:
            logger.error(f"Groq API Error during persona assessment ({persona}): {e}")
            return None

    def run_evaluation(self):
        json_files = glob.glob(os.path.join(self.input_dir, "*_eval_scripts.json"))

        if not json_files:
            logger.error(
                f"No generation files found in {self.input_dir}. Run D_generation.py first."
            )
            return

        for file_path in json_files:
            model_name = os.path.basename(file_path).replace("_eval_scripts.json", "")
            logger.info(f"=== Starting Multi-Persona Evaluation for Model: {model_name} ===")

            with open(file_path, "r", encoding="utf-8") as f:
                generation_data = json.load(f)

            evaluation_results = {}

            for topic, data in generation_data.items():
                script_id = data["id"]
                generated_script = data.get(model_name)

                if not generated_script or generated_script == "ERROR":
                    logger.warning(f"Skipping empty or failed script for topic: {topic}")
                    continue

                logger.info(f"Evaluating sketch: {script_id}...")

                raw_persona_evals = []
                metrics_accumulator = {
                    m: 0.0
                    for m in [
                        "novelty",
                        "clarity",
                        "relevance",
                        "intelligence",
                        "empathy",
                        "cultural_resonance",
                        "overall_score",
                    ]
                }

                for persona in tqdm(PERSONAS, desc=f"Personas judging '{script_id}'"):
                    eval_output = self.evaluate_with_persona(persona, topic, generated_script)

                    if eval_output:
                        eval_output["persona"] = persona
                        raw_persona_evals.append(eval_output)

                        for metric in metrics_accumulator.keys():
                            if metric in eval_output.keys():
                                metrics_accumulator[metric] += float(eval_output.get(metric, 0))
                            else:
                                print(eval_output)
                                logger.error(f'Failed to get "{metric}" metric for {script_id}')
                                continue

                total_valid_judges = len(raw_persona_evals)
                if total_valid_judges != len(PERSONAS):
                    logger.error(f"Failed to gather some valid persona assessments for {script_id}")
                    continue

                averaged_scores = {
                    metric: round(total / total_valid_judges, 2)
                    for metric, total in metrics_accumulator.items()
                }

                evaluation_results[topic] = {
                    "id": script_id,
                    "averaged_metrics": averaged_scores,
                    "raw_judgments": raw_persona_evals,
                }

            output_path = os.path.join(self.output_dir, f"{model_name}_final_metrics_api.json")
            with open(output_path, "w", encoding="utf-8") as out_f:
                json.dump(evaluation_results, out_f, ensure_ascii=False, indent=4)

            logger.info(f"Successfully saved evaluation matrix to: {output_path}")


if __name__ == "__main__":
    evaluator = ComedyScriptEvaluator()
    evaluator.run_evaluation()
