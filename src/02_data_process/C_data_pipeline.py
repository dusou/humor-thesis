import argparse
from dotenv import load_dotenv
import gc
import json
import logging
from omegaconf.base import ContainerMetadata
from omegaconf.dictconfig import DictConfig
from omegaconf.listconfig import ListConfig
import os
import pandas as pd
from pathlib import Path
import re
import shlex
import shutil
import torch
from typing import Any
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchaudio.*")
warnings.filterwarnings("ignore", message=".*TorchCodec.*")
warnings.filterwarnings("ignore", message=".*list_audio_backends.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

# Force Lightning to only show critical errors, hiding the "upgraded checkpoint" INFO logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning.pytorch.utilities.migration.utils").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)

# Tell PyTorch 2.6 to trust the VAD metadata used by WhisperX
torch.serialization.add_safe_globals([ListConfig, DictConfig, ContainerMetadata, Any])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()

# Connection to internet logic
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
hf_cache = Path(os.path.expanduser("~/.cache/huggingface/hub"))
qwen_folder = hf_cache / "models--Qwen--Qwen3.5-9B"

if qwen_folder.exists():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    logger.info("Local models found. Engaging offline mode.")
else:
    logger.info("Models missing. Allowing internet access for initial download...")

import demucs.separate
import transformers
import whisperx

transformers.logging.set_verbosity_error()

# Patch loads
original_load = torch.load


def patched_load(*args, **kwargs):
    kwargs["weights_only"] = False
    return original_load(*args, **kwargs)


import huggingface_hub

torch.load = patched_load

# Patch downloads
_original_download = huggingface_hub.hf_hub_download


def _patched_download(*args, **kwargs):
    if "use_auth_token" in kwargs:
        kwargs["token"] = kwargs.pop("use_auth_token")
    return _original_download(*args, **kwargs)


huggingface_hub.hf_hub_download = _patched_download


class LusoLaughDatasetGenerator:
    """
    Master pipeline for constructing the Luso-Laugh computational humor corpus.
    """

    def __init__(
        self,
        output_dir="../../data/02_audio_corpus",
        output_json_dir="../../data/03_final_dataset",
    ):
        self.output_dir = output_dir
        self.output_json_dir = output_json_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_json_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError("HF_TOKEN environment variable is missing. Check your .env file.")

        self._init_models()

    def _init_models(self):
        logger.info(f"Verified environment. Neural models will be loaded dynamically on {self.device.upper()}.")

        logger.info("Loading MIT AudioSet Transformer for laughter detection...")
        try:
            self.laughter_pipeline = transformers.pipeline(
                "audio-classification",
                model="MIT/ast-finetuned-audioset-10-10-0.4593",
                device=0 if self.device == "cuda" else -1,
            )
        except Exception as e:
            logger.error(f"Failed to load laughter model: {e}")
            self.laughter_pipeline = None

        logger.info("Loading Qwen3.5-9B...")
        try:
            self.llm_pipeline = transformers.pipeline(
                "text-generation",
                model="Qwen/Qwen3.5-9B",
                dtype=torch.bfloat16,
                device_map="auto",
            )
        except Exception as e:
            logger.error(f"Failed to load local LLM: {e}")
            self.llm_pipeline = None

    def separate_sources(self, audio_path: str, sketch_id: str) -> dict:
        out_dir = os.path.join(self.output_dir, sketch_id, "demucs_output")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        stem_dir = os.path.join(out_dir, "htdemucs", base_name)

        logger.info(f"Executing HTDemucs source separation for {sketch_id}...")
        cmd = f'--two-stems vocals -n htdemucs -j 8 --out "{out_dir}" "{audio_path}"'
        demucs.separate.main(shlex.split(cmd))

        return {
            "vocals": os.path.join(stem_dir, "vocals.wav"),
            "accompaniment": os.path.join(stem_dir, "no_vocals.wav"),
        }

    def detect_laughter(self, audio_path: str, chunk_duration=3.0, step_duration=2.0) -> list:
        logger.info("Running AI Audio Classification for laughter detection...")
        try:
            y = whisperx.load_audio(audio_path)
            sr = 16000

            chunk_samples = int(chunk_duration * sr)
            step_samples = int(step_duration * sr)
            raw_laughs = []

            laugh_labels = ["Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle"]

            # Overlapping Sliding Window
            # Grabs 3 seconds of audio, but only moves forward 2 seconds each time
            for i in range(0, len(y), step_samples):
                chunk = y[i : i + chunk_samples]

                if len(chunk) < sr:
                    continue

                # Top 20 detections
                result = self.laughter_pipeline(chunk, top_k=20)

                # Check if any of the laughter categories are in the top 20 with at least 5% confidence
                is_laugh = any(pred["label"] in laugh_labels and pred["score"] > 0.05 for pred in result)

                if is_laugh:
                    start_time = i / sr
                    end_time = (i + len(chunk)) / sr
                    raw_laughs.append({"start": float(start_time), "end": float(end_time)})

            # Merge overlapping 3-second chunks into continuous events
            merged_laughs = []
            for laugh in raw_laughs:
                if not merged_laughs:
                    merged_laughs.append(laugh)
                else:
                    last = merged_laughs[-1]
                    if laugh["start"] - last["end"] <= 2:
                        last["end"] = max(last["end"], laugh["end"])
                    else:
                        merged_laughs.append(laugh)

            logger.info(f"Detected {len(merged_laughs)} discrete laughter events.")
            return merged_laughs

        except Exception as e:
            logger.error(f"Laughter detection failed: {e}")
            return []

    def transcribe_and_diarize(self, vocals_path: str) -> list:
        logger.info("Executing WhisperX Transcription...")
        audio = whisperx.load_audio(vocals_path)

        # Transcribe
        model = whisperx.load_model("large-v3", self.device, compute_type=self.compute_type, language="pt")
        result = model.transcribe(audio, batch_size=16, language="pt")

        # Free memory
        del model
        gc.collect()
        torch.cuda.empty_cache()

        # Align
        logger.info(f"Aligning word-level timestamps for language: {result['language']}...")
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
        result = whisperx.align(result["segments"], model_a, metadata, audio, self.device, return_char_alignments=False)

        # Free memory
        del model_a
        gc.collect()
        torch.cuda.empty_cache()

        # Diarize
        logger.info("Executing Speaker Diarization...")
        diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=self.hf_token, device=self.device)
        diarize_segments = diarize_model(audio)

        # Free memory
        del diarize_model
        gc.collect()
        torch.cuda.empty_cache()

        # Merge
        logger.info("Assigning precise timestamps to speakers...")
        final_result = whisperx.assign_word_speakers(diarize_segments, result)

        # Format to match corpus schema
        aligned_script = []
        for segment in final_result["segments"]:
            aligned_script.append(
                {
                    "speaker": segment.get("speaker", "UNKNOWN"),
                    "text": segment["text"].strip(),
                    "start": segment["start"],
                    "end": segment["end"],
                }
            )

        return aligned_script

    def annotate_irony(self, aligned_script: list, laughs: list) -> list:
        logger.info("Starting semantic annotaion with LLM...")

        last_punchline_idx = 0

        for i, line in enumerate(aligned_script):
            line["is_punchline"] = False
            line["semantic_metadata"] = {}

            # Map laughs: If a laugh happens within 1 second of this line ending
            for laugh in laughs:
                if line["start"] <= laugh["start"] <= (line["end"] + 2):
                    line["is_punchline"] = True
                    break

            # If it is a punchline, prompt LLM for explanation
            if line["is_punchline"]:
                context_lines = aligned_script[last_punchline_idx:i]

                context_list = []
                for ctx_line in context_lines:
                    speaker = ctx_line.get("speaker", "UNKNOWN")
                    text = ctx_line.get("text", "")
                    context_list.append(f"[{speaker}]: {text}")

                context = "\n".join(context_list)
                current_speaker = line.get("speaker", "UNKNOWN")

                prompt = f"""
                És um analista linguístico profissional a estudar comédia em português.
                Abaixo está o contexto do diálogo, seguido pela piada final (punchline).
                --- DIALOGUE CONTEXT ---
                {context}
                --- TARGET PUNCHLINE ---
                [{current_speaker}]: "{line["text"]}"
                ------------------------
                Explica brevemente a ironia, o sarcasmo ou o humor da piada final com base no contexto.
                Responde em Português de Portugal. Máximo de 4 frases.
                REGRA CRÍTICA: Usa apenas vocabulário académico, formal e eufemismos educados na tua análise.

                FORMATO OBRIGATÓRIO:
                <raciocinio>
                [O raciocínio interno que motiva a tua explicação em menos de 10 frases]
                </raciocinio>
                <explicacao>
                [A tua síntese formal do humor, no máximo 5 frases]
                </explicacao>
                """

                messages = [
                    {
                        "role": "system",
                        "content": "You are a highly professional, polite, and academic AI analyzing Portuguese comedy. You strictly avoid profanity, slang, and vulgarity.",
                    },
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": "<raciocinio>\n"},
                ]

                try:
                    outputs = self.llm_pipeline(
                        messages,
                        temperature=0.5,
                        max_new_tokens=2084,
                        do_sample=True,
                        continue_final_message=True,
                    )

                    raw_text = outputs[0]["generated_text"][-1]["content"].strip()
                    raw_text = f"<raciocinio>\n{raw_text}"

                    match = re.search(r"<explicacao>(.*?)</explicacao>", raw_text, re.DOTALL | re.IGNORECASE)

                    if match:
                        final_explanation = match.group(1).strip()
                    else:
                        logger.warning(
                            f"XML tags missing in LLM output for line at {line['start']}s. Raw text: {raw_text[:50]}"
                        )
                        # Safe fallback: take the whole string but try to strip out the raciocinio part if it exists
                        final_explanation = re.sub(
                            r"<raciocinio>.*?</raciocinio>", "", raw_text, flags=re.DOTALL | re.IGNORECASE
                        ).strip()

                    # Clean up any leftover markdown bolding the model might have added
                    final_explanation = final_explanation.replace("**", "")

                    line["semantic_metadata"]["humor_analysis"] = final_explanation
                    logger.info(
                        f"\033[95mAnnotated punchline at {line['start']:.2f}s: {final_explanation[:50]}...\033[0m"
                    )

                    last_punchline_idx = i + 1

                except Exception as e:
                    line["semantic_metadata"]["humor_analysis"] = f"Local LLM Error: {str(e)}"
                    logger.warning(f"Failed to generate annotation for line at {line['start']:.2f}s.")

        return aligned_script

    def process_sketch_remote(self, sketch_id: str):
        audio_path_mp3 = os.path.join(self.output_dir, "{}.mp3".format(sketch_id))
        audio_path_wav = os.path.join(self.output_dir, "{}.wav".format(sketch_id))

        audio_path = audio_path_mp3 if os.path.exists(audio_path_mp3) else audio_path_wav

        if not os.path.exists(audio_path):
            logger.error(f"Audio file missing for {sketch_id}. Skipping.")
            return False

        logger.info(f"Commencing processing for {sketch_id}...")

        try:
            # stems = self.separate_sources(audio_path, sketch_id)
            laughs = self.detect_laughter(audio_path)
            script = self.transcribe_and_diarize(audio_path)
            final_corpus_entry = self.annotate_irony(script, laughs)

            out_file = os.path.join(self.output_json_dir, f"{sketch_id}_annotated.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final_corpus_entry, f, ensure_ascii=False, indent=4)

            logger.info(f"Luso-Laugh corpus entry serialized to {out_file}")

            sketch_demucs_folder = os.path.join(self.output_dir, sketch_id)
            if os.path.isdir(sketch_demucs_folder):
                shutil.rmtree(sketch_demucs_folder)

            return True

        except Exception as e:
            logger.error(f"Pipeline crashed for {sketch_id}: {e}")
            return False


# === Execution Loop ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LusoLaugh Remote Orchestrator")
    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    catalog_file = os.path.join(dirname, "../../data/01_catalogs/luso_laugh_catalog.csv")

    if not os.path.exists(catalog_file):
        logger.error(f"Catalog file '{catalog_file}' not found.")
        exit()

    df = pd.read_csv(catalog_file)

    ready_sketches = df[df["status"] == "downloaded"]
    logger.info(f"Found {len(ready_sketches)} sketches ready for AI processing.")

    output_dir = os.path.normpath(os.path.join(dirname, "../../data/02_audio_corpus/"))
    output_json_dir = os.path.normpath(os.path.join(dirname, "../../data/03_final_dataset/"))

    generator = LusoLaughDatasetGenerator(output_dir=output_dir, output_json_dir=output_json_dir)

    total_sketches = len(ready_sketches)

    for current_count, (index, row) in enumerate(ready_sketches.iterrows(), start=1):
        sketch_id = str(row["sketch_id"])
        logger.info(f"\033[96m--- Processing Sketch ID: {sketch_id} ({current_count}/{total_sketches}) ---\033[0m")

        success = generator.process_sketch_remote(sketch_id)

        if success:
            df.at[index, "status"] = "processed"
            df.to_csv(catalog_file, index=False)
