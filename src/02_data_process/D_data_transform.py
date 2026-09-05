import argparse
import json
import logging
import os
import pandas as pd
from pathlib import Path
import re
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


SYSTEM_PROMPT = (
    "És um argumentista profissional de comédia e sátira portuguesa.\n"
    "Escreves exclusivamente em Português Europeu (PT-PT), usando o vocabulário, "
    "a sintaxe e as expressões idiomáticas correntes em Portugal.\n"
    "O teu humor é observacional, irónico e subversivo: ancoras as piadas na "
    "realidade social, política e quotidiana portuguesa e escalas o absurdo a "
    "partir de premissas reconhecíveis.\n"
    "Nunca explicas a piada depois de a fazeres e "
    "preferes o risco cómico à segurança de um texto genérico.\n"
    "Se te forem dados sketches de referência, usa-os apenas como modelo de "
    "ritmo, cadência e registo, nunca reaproveites as suas falas ou premissas."
)

MACRO_INSTRUCTION = (
    "Escreve um sketch de comédia original em Português de Portugal a partir do "
    "tema e premissa indicados no fim.\n\n"
    "Antes de escreveres, planeia o arco cómico completo seguindo a estrutura: "
    "ELENCO, ABORDAGEM REJEITADA, ARCO CÓMICO (com expectativa, violação e "
    "lógica interna para cada piada) e ESCALADA. Extensão do plano: 25 a 30 frases.\n\n"
    "Regras de escrita:\n"
    "1. Escreve o guião em falas. Cada fala ocupa uma linha própria, precedida "
    "pelo nome da personagem em maiúsculas entre parênteses retos: [NOME]: fala.\n"
    "2. Podes acrescentar didascálias breves em linha própria, entre parênteses "
    "retos e sem dois pontos, no máximo uma por cada seis falas.\n"
    "3. Fixa as personagens no início e mantém-nas até ao fim.\n"
    "4. Constrói uma escalada: cada piada deve subir a aposta da anterior e "
    "terminar na punchline mais forte.\n"
    "5. Usa referências culturais portuguesas concretas em vez de genéricas.\n"
    "6. Extensão alvo: 500 a 900 palavras.\n\n"
    "Tema e premissa:"
)


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

    def _merge_turns(self, data: list) -> list:
        """Collapse consecutive segments from the same speaker so one line is one turn."""
        turns = []
        for line in data:
            spk = line.get("speaker", "UNKNOWN")
            txt = (line.get("text") or "").strip()
            if not txt:
                continue
            if turns and turns[-1][0] == spk:
                turns[-1][1] += " " + txt
            else:
                turns.append([spk, txt])
        return turns

    def _generate_script(self, sketch_id: str, transcript: str) -> str:
        """Convert a diarised transcript into a script with named characters
        and sparse stage directions. Falls back to the raw transcript on any
        anomaly, so downstream code always receives a valid script."""
        if self.dry_run or not self.llm_pipeline:
            return transcript

        original_turns = [
            (m.group(1), m.group(2))
            for m in (re.match(r"^\[([^\]]+)\]:\s*(.*)$", ln.strip()) for ln in transcript.split("\n"))
            if m
        ]
        if not original_turns:
            return transcript

        speakers = list({s for s, _ in original_turns})
        first_line_text = original_turns[0][1]

        prompt = (
            "Abaixo está a transcrição diarizada de um sketch de comédia portuguesa.\n"
            f"Etiquetas de interveniente presentes: {', '.join(speakers)}.\n\n"
            "TAREFA\n"
            "Reescreve a transcrição como guião, aplicando exactamente estas transformações:\n"
            "1. Antes de reescrever, decide um nome para CADA etiqueta que aparece na "
            "transcrição. Escreve essa correspondência primeiro, no formato:\n"
            "   SPEAKER_00 -> NOME1\n"
            "   SPEAKER_01 -> NOME2\n"
            "   (uma linha por cada etiqueta distinta)\n"
            "   Depois escreve o guião usando esses nomes. NUNCA dês o mesmo nome a "
            "etiquetas diferentes.\n"
            "2. Formata cada fala como uma linha, no formato exacto:\n"
            "   [NOME]: fala\n"
            "3. Onde o conteúdo o implicar de forma inequívoca, podes acrescentar didascálias muito breves "
            "em linha própria, entre parênteses retos, sem dois pontos, ex.:\n"
            "   [Olha para a fatura com horror.]\n"
            "   No máximo uma didascália por cada seis falas.\n\n"
            "REGRAS ABSOLUTAS\n"
            "- Preserva LITERALMENTE cada palavra do texto original, pela mesma ordem. "
            "Não reescrevas, não reformules, não corrijas gramática, não substituas sinónimos.\n"
            "- Podes DIVIDIR uma fala longa em várias linhas se ela contiver claramente "
            "várias intervenções separadas (por exemplo, um narrador que muda de tópico, "
            "ou momentos em que o mesmo interveniente muda de registo).\n"
            "- Podes acrescentar didascálias breves em linha própria, entre parênteses "
            "retos e sem dois pontos.\n"
            "- Não inventes acontecimentos que o texto não implique.\n"
            f"--- TRANSCRIÇÃO ({len(original_turns)} falas) ---\n{transcript}\n--- FIM ---"
        )

        messages = [
            {
                "role": "system",
                "content": "És um argumentista que converte transcrições em guiões, preservando cada fala.",
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "["},
        ]

        try:
            outputs = self.llm_pipeline(
                messages,
                temperature=0.3,
                max_new_tokens=4096,
                do_sample=True,
                continue_final_message=True,
            )
            raw = "[" + outputs[0]["generated_text"][-1]["content"]
        except Exception as e:
            logger.warning(f"{sketch_id}: script enrichment call failed ({e}). Using transcript.")
            return transcript

        script = self._sanitize_script(raw)

        # Line-count guard. Model must have kept roughly all the turns.
        script_turns = [l for l in script.split("\n") if re.match(r"^\[[A-ZÁÂÃÀÇÉÊÍÓÔÕÚ][^\]]*\]:\s*.+", l.strip())]
        n_in, n_out = len(original_turns), len(script_turns)
        input_chars = len(transcript)
        output_chars = sum(len(t) for t in script_turns)
        if output_chars < 0.60 * input_chars or output_chars > 1.40 * input_chars:
            logger.warning(
                f"{sketch_id}: script content changed by {output_chars / input_chars:.0%} "
                f"({input_chars} -> {output_chars} chars). Using transcript."
            )
            return transcript
        if n_out < n_in:
            logger.warning(f"{sketch_id}: script lost turns ({n_in} -> {n_out}). Using transcript.")
            return transcript

        # Format regression guard. Did the model reintroduce SPEAKER_NN tags?
        if re.search(r"\[SPEAKER_\d+\]", script):
            logger.warning(f"{sketch_id}: script still contains SPEAKER_NN tags. Using transcript.")
            return transcript

        logger.info(f"Enriched {sketch_id}: {n_in} turns -> {n_out} lines")
        return script

    def _sanitize_script(self, raw: str) -> str:
        """Strip markdown the model sometimes emits."""
        text = raw.strip()
        text = re.sub(r"^\[\[", "[", text, flags=re.MULTILINE)  # collapse double brackets
        text = re.sub(r"\*\*", "", text)  # bold markers
        text = re.sub(r"^```.*?\n", "", text)  # opening code fence
        text = re.sub(r"\n```$", "", text)  # closing code fence
        # Drop any leading lines before the first line
        lines = text.split("\n")
        for i, l in enumerate(lines):
            if re.match(r"^\[[A-ZÁÂÃÀÇÉÊÍÓÔÕÚ][^\]]*\]", l.strip()):
                return "\n".join(lines[i:]).strip()
        return text

    def _render_script(self, data: list) -> str:
        return "\n".join(f"[{s}]: {t}" for s, t in self._merge_turns(data))

    def format_for_rag(self, sketch_id: str, enriched_script: str, data: list):
        """
        Aggregates the sketch into a single document, appending all successful
        humor analyses to provide semantic anchors for the vector database.
        """
        analyses = []
        for line in data:  # segment-level, punchline flags intact
            if line.get("is_punchline", False):
                analysis = line.get("semantic_metadata", {}).get("humor_analysis")
                if analysis and "Local LLM Error" not in analysis:
                    text = line.get("text", "")
                    analyses.append(f"Analysis of '{text}': {analysis}")

        rag_document = {
            "sketch_id": sketch_id,
            "content": enriched_script,
            "comedic_metadata": "\n".join(analyses),
        }

        out_path = self.rag_output_dir / f"{sketch_id}_rag.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(rag_document, f, ensure_ascii=False, indent=4)

    def _generate_synthetic_summary(self, sketch_id: str, script: str) -> str:
        fallback_summary = "Um sketch de comédia portuguesa."

        if self.dry_run or not self.llm_pipeline:
            return fallback_summary

        prompt = (
            "Lê a seguinte transcrição de um texto de comédia portuguesa.\n"
            f"--- TRANSCRIÇÃO ---\n{script}\n--------------------\n\n"
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
            "Antes de escreveres o sketch, planeia em voz alta, na primeira pessoa e no FUTURO, "
            "o arco cómico COMPLETO. Segue esta estrutura:\n\n"
            "1. ELENCO\n"
            "   Para cada personagem, indica um nome curto ou papel e uma "
            "característica de voz que a distinga das outras (registo, tique verbal, "
            "obsessão). Mantém esse elenco fixo durante todo o plano.\n\n"
            "2. ABORDAGEM REJEITADA\n"
            "   Considera brevemente uma abordagem óbvia para esta premissa e "
            "explica em uma ou duas frases por que a vais rejeitar por ser previsível "
            "ou por cair em clichê.\n\n"
            "3. ARCO CÓMICO\n"
            "   Descreve a abordagem que vais efectivamente usar e como abres a cena. "
            "Depois, para CADA piada da escalada, escreve pelo menos três frases:\n"
            "     (a) uma frase que descreva a expectativa concreta que a montagem cria no espectador;\n"
            "     (b) uma frase que descreva o elemento específico que viola essa expectativa;\n"
            "     (c) uma frase que explique a lógica interna que torna a violação compreensível em vez de arbitrária.\n\n"
            "4. ESCALADA E PUNCHLINE\n"
            "   Explica como cada piada eleva a aposta da anterior e porque é que a "
            "punchline final é o ponto de maior distância entre expectativa e desfecho.\n\n"
            "REGRAS:\n"
            "- NÃO expliques piadas isoladas fora deste plano, sintetiza tudo num ÚNICO plano coeso.\n"
            "- Sê estruturado. Extensão alvo: entre 25 e 30 frases.\n\n"
            "FORMATO OBRIGATÓRIO:\nPLANO: [o teu plano aqui]"
        )

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": "PLANO: Vou"},
        ]

        try:
            outputs = self.llm_pipeline(
                messages,
                temperature=0.7,
                max_new_tokens=3500,
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

    def format_for_lora(self, sketch_id: str, data: list, enriched_script: str, lora_file) -> int:
        punchline_beats = []
        for line in data:
            if line.get("is_punchline", False):
                analysis = line.get("semantic_metadata", {}).get("humor_analysis")
                if analysis and "Local LLM Error" not in analysis:
                    punchline_beats.append((line.get("speaker", "UNKNOWN"), line.get("text", ""), analysis))

        variants = 1 if self.dry_run else self.variants
        macro_count = 0

        for _ in range(variants):
            summary_input = self._generate_synthetic_summary(sketch_id, enriched_script)
            reasoning = self._generate_synthetic_arc_reasoning(sketch_id, summary_input, punchline_beats)
            macro_output = f"<think>\n{reasoning}\n</think>\n\n{enriched_script}"

            lora_entry = {
                "task": "macro",
                "sketch_id": sketch_id,
                "instruction": MACRO_INSTRUCTION,
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

        transcript = self._render_script(data)
        enriched_script = self._generate_script(sketch_id, transcript)

        with open(self.lora_output_file, "a", encoding="utf-8") as lora_f:
            macro_count = self.format_for_lora(sketch_id, data, enriched_script, lora_f)
        self.format_for_rag(sketch_id, enriched_script, data)

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
    parser.add_argument("--limit", type=int, default=None, help="Process at most N sketches this run (testing).")
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent.parent / "data"
    input_dir = base_dir / "03_final_dataset"
    rag_output_dir = base_dir / "04_rag_ready"
    lora_output_dir = base_dir / "04_lora_ready"

    input_dir.mkdir(parents=True, exist_ok=True)
    rag_output_dir.mkdir(parents=True, exist_ok=True)
    lora_output_dir.mkdir(parents=True, exist_ok=True)

    catalog_path = base_dir / "01_catalogs" / "luso_laugh_catalog.csv"
    if not catalog_path.exists():
        logger.error(f"Catalog file '{catalog_path}' not found. Cannot filter scripts.")
        exit(1)

    df_catalog = pd.read_csv(catalog_path)
    valid_sketch_ids = set(df_catalog["sketch_id"].astype(str))

    input_files = list(input_dir.glob("*_annotated.json"))
    processed_ids = {f.stem.replace("_rag", "") for f in rag_output_dir.glob("*_rag.json")}
    pending_files = [
        f
        for f in input_files
        if f.stem.replace("_annotated", "") in valid_sketch_ids
        and f.stem.replace("_annotated", "") not in processed_ids
    ]

    logger.info("=" * 45)
    logger.info("LUSO-LAUGH TRANSFORM PIPELINE STATUS")
    logger.info("=" * 45)
    logger.info(f"Total valid annotated sketches:    {len(valid_sketch_ids)}")
    logger.info(f"Total annotated sketches:    {len(input_files)}")
    logger.info(f"Already transformed:         {len(processed_ids)}")
    logger.info(f"To process this run:         {len(pending_files)}")
    logger.info(f"Variants per sketch:         {args.macro_variants}")
    logger.info("=" * 45)

    if args.limit:
        pending_files = pending_files[: args.limit]
        logger.info(f"Limiting this run to {len(pending_files)} sketches.")

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
