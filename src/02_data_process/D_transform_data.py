import json
import logging
import os
from pathlib import Path
import torch
import transformers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce transformer warnings
transformers.logging.set_verbosity_error()

os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
qwen_folder = hf_cache / "models--Qwen--Qwen3.5-9B"

if qwen_folder.exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("Local models found. Engaging offline mode.")
else:
    logger.info("Models missing. Allowing internet access for initial download...")


class TrainingDataFormatter:
    """
    Transforms the Luso-Laugh annotated JSON files into RAG-ready documents
    and a multi-task LoRA JSONL dataset (Micro/Macro comedy generation).
    """

    def __init__(
        self,
        input_dir="../../data/03_final_dataset",
        rag_output_dir="../../data/04_rag_ready",
        lora_output_dir="../../data/04_lora_ready",
        dry_run=False,
    ):
        self.input_dir = Path(input_dir)
        self.rag_output_dir = Path(rag_output_dir)
        self.lora_output_dir = Path(lora_output_dir)
        self.dry_run = dry_run

        # Ensure output directories exist
        self.rag_output_dir.mkdir(parents=True, exist_ok=True)
        self.lora_output_dir.mkdir(parents=True, exist_ok=True)

        self.lora_output_file = self.lora_output_dir / "lora_instruction_dataset.jsonl"

        self.hf_token = os.getenv("HF_TOKEN")

        self._init_llm()

    def _init_llm(self):
        if self.dry_run:
            logger.info("DRY RUN: Skipping LLM pipeline initialization.")
            self.llm_pipeline = None
            return

        logger.info("Loading Qwen3.5-9B for synthetic summarization...")
        try:
            self.llm_pipeline = transformers.pipeline(
                "text-generation",
                model="Qwen/Qwen3.5-9B",
                dtype=torch.bfloat16,
                device_map="auto",
                token=self.hf_token,
            )
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            self.llm_pipeline = None

    def format_for_rag(self, sketch_id: str, data: list):
        """
        Aggregates the sketch into a single document, appending all successful
        humor analyses to provide semantic anchors for the vector database.
        """
        full_text_lines = []
        analyses = []

        for line in data:
            speaker = line.get("speaker", "UNKNOWN")
            text = line.get("text", "")
            full_text_lines.append(f"[{speaker}]: {text}")

            if line.get("is_punchline", False):
                analysis = line.get("semantic_metadata", {}).get("humor_analysis")
                if analysis and "Local LLM Error" not in analysis:
                    analyses.append(f"Analysis of '{text}': {analysis}")

        rag_document = {
            "sketch_id": sketch_id,
            "content": "\n".join(full_text_lines),
            "comedic_metadata": "\n".join(analyses),
        }

        out_path = self.rag_output_dir / f"{sketch_id}_rag.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rag_document, f, ensure_ascii=False, indent=4)

    def _generate_synthetic_summary(self, sketch_id: str, data: list) -> str:
        """
        Uses the local model to generate a short, structural premise
        for the given sketch transcript to be used in the Macro LoRA task.
        """
        speakers = list(set([line.get("speaker", "UNKNOWN") for line in data]))
        fallback_summary = (
            f"Um sketch de comédia portuguesa envolvendo uma interação entre {', '.join(speakers)}."
        )

        if self.dry_run or not self.llm_pipeline:
            return fallback_summary

        transcript_lines = [
            f"[{line.get('speaker', 'UNKNOWN')}]: {line.get('text', '')}" for line in data
        ]
        full_transcript = "\n".join(transcript_lines)

        prompt = f"""
                Lê a seguinte transcrição de um texto de comédia portuguesa. Este texto poderá ser um sketch, um programa de televisão, um podcast ou outro meio de difusão de comédia.

                --- TRANSCRIÇÃO ---
                {full_transcript}
                --------------------

                Escreve um breve resumo (5 frases no máximo) que descreva a premissa principal, o cenário e a dinâmica deste texto. 
                Responde estritamente em Português de Portugal.

                FORMATO OBRIGATÓRIO:
                RESUMO: [O teu resumo aqui]
                """

        messages = [
            {
                "role": "system",
                "content": "És um argumentista profissional e analista de comédia. És conciso, direto e captas perfeitamente a essência de uma cena.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "RESUMO:"},
        ]

        try:
            outputs = self.llm_pipeline(
                messages,
                temperature=0.7,
                max_new_tokens=1024,
                do_sample=True,
                continue_final_message=True,
            )

            raw_text = outputs[0]["generated_text"][-1]["content"].strip()

            final_summary = raw_text.replace("RESUMO:", "").replace("**", "").strip()

            logger.info(f"Generated summary for {sketch_id}: {final_summary[:60]}...")
            return final_summary

        except Exception as e:
            logger.warning(f"LLM summarization failed: {e}. Reverting to fallback summary.")
            return fallback_summary

    def format_for_lora(self, sketch_id: str, data: list, lora_file):
        """
        Routes the sketch to either the Micro task (punchline completion)
        or the Macro task (full sketch generation from premise) based on laughter presence.
        """
        has_punchlines = any(line.get("is_punchline", False) for line in data)

        if has_punchlines:
            # ==========================================
            # TASK A: MICRO (Chain-of-Thought Punchlines)
            # ==========================================
            instruction = (
                "Continue the following Portuguese comedy sketch. First, outline the comedic "
                "logic and ironic subtext you will use. Then, generate the exact dialogue for the next punchline."
            )

            for i, line in enumerate(data):
                if line.get("is_punchline", False):
                    analysis = line.get("semantic_metadata", {}).get("humor_analysis")

                    if not analysis or "Local LLM Error" in analysis:
                        continue

                    # Cumulative context from the very beginning up to the punchline
                    context_lines = data[:i]
                    if not context_lines:
                        continue

                    context_str = "\n".join(
                        [
                            f"[{c.get('speaker', 'UNKNOWN')}]: {c.get('text', '')}"
                            for c in context_lines
                        ]
                    )
                    current_speaker = line.get("speaker", "UNKNOWN")
                    punchline_text = line.get("text", "")

                    output_str = (
                        f"[Raciocínio] {analysis} [Punchline] [{current_speaker}]: {punchline_text}"
                    )

                    lora_entry = {
                        "instruction": instruction,
                        "input": context_str,
                        "output": output_str,
                    }
                    lora_file.write(json.dumps(lora_entry, ensure_ascii=False) + "\n")

        else:
            # ==========================================
            # TASK B: MACRO (Context Expansion)
            # ==========================================
            instruction = "Write a complete Portuguese comedy sketch based on the following premise and stylistic direction."

            summary_input = self._generate_synthetic_summary(sketch_id, data)
            full_transcript = "\n".join(
                [f"[{line.get('speaker', 'UNKNOWN')}]: {line.get('text', '')}" for line in data]
            )

            lora_entry = {
                "instruction": instruction,
                "input": summary_input,
                "output": full_transcript,
            }
            lora_file.write(json.dumps(lora_entry, ensure_ascii=False) + "\n")

    def run(self):
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} not found.")
            return

        json_files = list(self.input_dir.glob("*_annotated.json"))
        logger.info(f"Found {len(json_files)} annotated sketches to process.")

        with open(self.lora_output_file, "w", encoding="utf-8") as lora_f:
            for filepath in json_files:
                sketch_id = filepath.name.replace("_annotated.json", "")

                with open(filepath, "r", encoding="utf-8") as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse {filepath.name}. Skipping.")
                        continue

                self.format_for_rag(sketch_id, data)
                self.format_for_lora(sketch_id, data, lora_f)

        logger.info("Transformation complete.")
        logger.info(f"RAG files saved to: {self.rag_output_dir}")
        logger.info(f"LoRA JSONL saved to: {self.lora_output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format LusoLaugh Data")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading the LLM")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    input_dir = os.path.normpath(os.path.join(dirname, "../../data/03_final_dataset/"))
    rag_output_dir = os.path.normpath(os.path.join(dirname, "../../data/04_rag_ready/"))
    lora_output_dir = os.path.normpath(os.path.join(dirname, "../../data/04_lora_ready/"))

    formatter = TrainingDataFormatter(
        dry_run=args.dry_run,
        input_dir=input_dir,
        rag_output_dir=rag_output_dir,
        lora_output_dir=lora_output_dir,
    )
    formatter.run()
