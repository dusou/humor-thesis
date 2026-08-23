import argparse
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
    Transforms Luso-Laugh annotated JSON files into RAG-ready documents
    and a LoRA JSONL dataset specialized in full-sketch generation.
    """

    def __init__(
        self,
        input_dir="../../data/03_final_dataset",
        rag_output_dir="../../data/04_rag_ready",
        lora_output_dir="../../data/04_lora_ready",
        dry_run=False,
        variants_per_sketch=1,
    ):
        self.input_dir = Path(input_dir)
        self.rag_output_dir = Path(rag_output_dir)
        self.lora_output_dir = Path(lora_output_dir)
        self.dry_run = dry_run
        self.variants = variants_per_sketch

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
        speakers = list({line.get("speaker", "UNKNOWN") for line in data})
        fallback_summary = f"Um sketch de comédia portuguesa envolvendo uma interação entre {', '.join(speakers)}."

        if self.dry_run or not self.llm_pipeline:
            return fallback_summary

        full_transcript = "\n".join([f"[{line.get('speaker', 'UNKNOWN')}]: {line.get('text', '')}" for line in data])

        prompt = (
            "Lê a seguinte transcrição de um texto de comédia portuguesa.\n"
            f"--- TRANSCRIÇÃO ---\n{full_transcript}\n--------------------\n\n"
            "Escreve um breve resumo (5 frases no máximo) que descreva a premissa principal, "
            "o cenário e a dinâmica deste texto.\n"
            "Responde estritamente em Português de Portugal.\n\n"
            "FORMATO OBRIGATÓRIO:\nRESUMO: [O teu resumo aqui]"
        )

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

    def _generate_synthetic_arc_reasoning(self, sketch_id: str, premise: str, punchline_beats: list) -> str:
        fallback_reasoning = (
            f"Vou escrever um sketch original com a seguinte premissa: {premise} "
            "Vou estruturar o texto com uma escalada gradual de absurdo, encadeando várias piadas até à punchline final."
        )

        if self.dry_run or not self.llm_pipeline:
            return fallback_reasoning

        reference_note = ""
        if punchline_beats:
            max_beats = 10
            if len(punchline_beats) > max_beats:
                step = len(punchline_beats) / max_beats
                sampled = [punchline_beats[int(i * step)] for i in range(max_beats)]
            else:
                sampled = punchline_beats

            beats_block = "\n".join(
                f"{i + 1}. (sobre '{text[:60]}...') {analysis}" for i, (speaker, text, analysis) in enumerate(sampled)
            )
            reference_note = (
                "Este sketch original continha esta sequência de piadas (analisadas a posteriori, "
                f"por ordem de aparição, apenas como inspiração):\n{beats_block}\n\n"
            )

        prompt = (
            f'Vais escrever um sketch de comédia portuguesa completo com a seguinte premissa:\n"{premise}"\n\n'
            f"{reference_note}"
            "Antes de escreveres o sketch, planeia em voz alta, na primeira pessoa e no FUTURO, o arco cómico COMPLETO do texto: "
            "como vais abrir a cena, que técnica de ironia/sátira vais usar em cada piada sucessiva, como escalam, e como termina a punchline final.\n"
            "NÃO expliques piadas isoladas -- sintetiza tudo num ÚNICO plano coeso de progressão.\n"
            "Sê estruturado mas conciso (máximo 8 frases).\n\n"
            "FORMATO OBRIGATÓRIO:\nPLANO: [o teu plano aqui]"
        )

        messages = [
            {
                "role": "system",
                "content": "És um argumentista profissional de comédia portuguesa a planear a estrutura completa de um novo sketch.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "PLANO:"},
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
            final_reasoning = raw_text.replace("PLANO:", "").replace("**", "").strip()

            logger.info(f"Generated arc reasoning for {sketch_id}: {final_reasoning[:60]}...")
            return final_reasoning
        except Exception as e:
            logger.warning(f"LLM arc reasoning synthesis failed: {e}. Reverting to fallback.")
            return fallback_reasoning

    def format_for_lora(self, sketch_id: str, data: list, lora_file) -> int:
        punchline_beats = []
        for line in data:
            if line.get("is_punchline", False):
                analysis = line.get("semantic_metadata", {}).get("humor_analysis")
                if analysis and "Local LLM Error" not in analysis:
                    punchline_beats.append((line.get("speaker", "UNKNOWN"), line.get("text", ""), analysis))

        macro_instruction = "Escreve um novo sketch de comédia sobre o seguinte tema e premissa:"
        full_transcript = "\n".join([f"[{line.get('speaker', 'UNKNOWN')}]: {line.get('text', '')}" for line in data])

        variants = 1 if self.dry_run else self.variants
        macro_count = 0

        for _ in range(variants):
            summary_input = self._generate_synthetic_summary(sketch_id, data)
            reasoning = self._generate_synthetic_arc_reasoning(sketch_id, summary_input, punchline_beats)
            macro_output = f"<think>\n{reasoning}\n</think>\n\n{full_transcript}"

            lora_entry = {
                "task": "macro",
                "sketch_id": sketch_id,
                "instruction": macro_instruction,
                "input": summary_input,
                "output": macro_output,
            }
            lora_file.write(json.dumps(lora_entry, ensure_ascii=False) + "\n")
            macro_count += 1

        return macro_count

    def process_sketch(self, filepath: Path) -> tuple[bool, int]:
        sketch_id = filepath.stem.replace("_annotated", "")

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse {filepath.name}. Skipping.")
            return False, 0

        self.format_for_rag(sketch_id, data)

        with open(self.lora_output_file, "a", encoding="utf-8") as lora_f:
            macro_count = self.format_for_lora(sketch_id, data, lora_f)

        return True, macro_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Format LusoLaugh Data")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading the LLM")
    parser.add_argument(
        "--macro-variants",
        type=int,
        default=1,
        help="Number of diverse full-sketch training examples to generate per sketch.",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    input_dir = base_dir / "03_final_dataset"
    rag_output_dir = base_dir / "04_rag_ready"
    lora_output_dir = base_dir / "04_lora_ready"

    input_dir.mkdir(parents=True, exist_ok=True)
    rag_output_dir.mkdir(parents=True, exist_ok=True)
    lora_output_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(input_dir.glob("*_annotated.json"))
    processed_ids = {f.stem.replace("_rag", "") for f in rag_output_dir.glob("*_rag.json")}
    pending_files = [f for f in input_files if f.stem.replace("_annotated", "") not in processed_ids]

    logger.info("=" * 45)
    logger.info("LUSO-LAUGH TRANSFORM PIPELINE STATUS")
    logger.info("=" * 45)
    logger.info(f"Total annotated sketches:    {len(input_files)}")
    logger.info(f"Already transformed:         {len(processed_ids)}")
    logger.info(f"To process this run:         {len(pending_files)}")
    logger.info(f"Variants per sketch:         {args.macro_variants}")
    logger.info("=" * 45)

    if not pending_files:
        logger.info("No sketches pending transformation. Exiting.")
        exit()

    formatter = TrainingDataFormatter(
        dry_run=args.dry_run,
        input_dir=input_dir,
        rag_output_dir=rag_output_dir,
        lora_output_dir=lora_output_dir,
        variants_per_sketch=args.macro_variants,
    )

    total_examples = 0
    sketches_processed = 0

    for current_count, filepath in enumerate(pending_files, start=1):
        sketch_id = filepath.stem.replace("_annotated", "")
        logger.info(
            f"\033[96m--- Transforming Sketch ID: {sketch_id} ({current_count}/{len(pending_files)}) ---\033[0m"
        )

        success, example_count = formatter.process_sketch(filepath)
        if success:
            total_examples += example_count
            sketches_processed += 1

    logger.info("Transformation complete.")
    logger.info(f"RAG files saved to: {rag_output_dir}")
    logger.info(f"LoRA JSONL saved to: {formatter.lora_output_file}")
    logger.info(f"Sketches processed this run:  {sketches_processed}")
    logger.info(f"Total LoRA examples generated:{total_examples}")

    if sketches_processed > 0:
        logger.info(f"Average examples per sketch:  {total_examples / sketches_processed:.1f}")
