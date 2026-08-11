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
        fallback_summary = f"Um sketch de comédia portuguesa envolvendo uma interação entre {', '.join(speakers)}."

        if self.dry_run or not self.llm_pipeline:
            return fallback_summary

        transcript_lines = [f"[{line.get('speaker', 'UNKNOWN')}]: {line.get('text', '')}" for line in data]
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
                max_new_tokens=2048,
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
                        [f"[{c.get('speaker', 'UNKNOWN')}]: {c.get('text', '')}" for c in context_lines]
                    )
                    current_speaker = line.get("speaker", "UNKNOWN")
                    punchline_text = line.get("text", "")

                    output_str = f"[Raciocínio] {analysis} [Punchline] [{current_speaker}]: {punchline_text}"

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
            instruction = (
                "Write a complete Portuguese comedy sketch based on the following premise and stylistic direction."
            )

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

    def process_sketch(self, filepath: Path):
        """Processes a single sketch and saves its RAG and LoRA representations."""
        sketch_id = filepath.name.replace("_annotated.json", "")

        with open(filepath, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse {filepath.name}. Skipping.")
                return False

        # RAG adaptation
        self.format_for_rag(sketch_id, data)

        # LORA JSONL adaptation
        with open(self.lora_output_file, "a", encoding="utf-8") as lora_f:
            self.format_for_lora(sketch_id, data, lora_f)

        return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Format LusoLaugh Data")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading the LLM")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    input_dir = Path(os.path.normpath(os.path.join(dirname, "../../data/03_final_dataset/")))
    rag_output_dir = Path(os.path.normpath(os.path.join(dirname, "../../data/04_rag_ready/")))
    lora_output_dir = Path(os.path.normpath(os.path.join(dirname, "../../data/04_lora_ready/")))

    input_dir.mkdir(parents=True, exist_ok=True)
    rag_output_dir.mkdir(parents=True, exist_ok=True)
    lora_output_dir.mkdir(parents=True, exist_ok=True)

    # Find all source files
    input_files = list(input_dir.glob("*_annotated.json"))

    # Check which ones have already been generated in the RAG folder
    processed_files = list(rag_output_dir.glob("*_rag.json"))
    processed_ids = set(f.name.replace("_rag.json", "") for f in processed_files)

    # Filter down to only what is missing
    pending_files = [f for f in input_files if f.name.replace("_annotated.json", "") not in processed_ids]

    total_count = len(input_files)
    processed_count = len(processed_ids)
    ready_count = len(pending_files)

    logger.info("=" * 45)
    logger.info("LUSO-LAUGH TRANSFORM PIPELINE STATUS")
    logger.info("=" * 45)
    logger.info(f"Total annotated sketches:\t{total_count}")
    logger.info(f"Already transformed:\t\t{processed_count}")
    logger.info(f"To process this run:\t\t{ready_count}")
    logger.info("=" * 45)

    if ready_count == 0:
        logger.info("No sketches are currently pending transformation. Exiting.")
        exit()

    formatter = TrainingDataFormatter(
        dry_run=args.dry_run,
        input_dir=str(input_dir),
        rag_output_dir=str(rag_output_dir),
        lora_output_dir=str(lora_output_dir),
    )

    for current_count, filepath in enumerate(pending_files, start=1):
        sketch_id = filepath.name.replace("_annotated.json", "")

        logger.info(f"\033[96m--- Transforming Sketch ID: {sketch_id} ({current_count}/{ready_count}) ---\033[0m")
        formatter.process_sketch(filepath)

    logger.info("Transformation complete.")
    logger.info(f"RAG files saved to: {rag_output_dir}")
    logger.info(f"LoRA JSONL saved to: {formatter.lora_output_file}")
