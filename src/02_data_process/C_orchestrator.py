import argparse
import demucs.separate
from dotenv import load_dotenv
from faster_whisper import WhisperModel
import json
import logging
import os
import pandas as pd
from pyannote.audio import Pipeline
from pyannote.audio.core.task import Problem, Resolution, Specifications
import shlex
import torch
from torch.torch_version import TorchVersion
import warnings

# Ignores some warnings from pyannote
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")
warnings.filterwarnings("ignore", message=".*degrees of freedom is <= 0.*")

# Tell PyTorch 2.6's security system to trust Pyannote's metadata
torch.serialization.add_safe_globals([TorchVersion, Specifications, Problem, Resolution])

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

load_dotenv()


class LusoLaughDatasetGenerator:
    """
    Master pipeline for constructing the Luso-Laugh computational humor corpus.
    Includes a dry-run flag for local path and logic testing.
    """

    def __init__(
        self,
        output_dir="../../data/02_audio_corpus",
        output_json_dir="../../data/03_final_dataset",
        dry_run=False,
    ):
        self.output_dir = output_dir
        self.output_json_dir = output_json_dir
        self.dry_run = dry_run

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.output_json_dir, exist_ok=True)

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self._init_models()

    def _init_models(self):
        # IF DRY RUN: Skip all heavy model initializations
        if self.dry_run:
            logger.info("DRY RUN MODE ACTIVE: Skipping neural pipeline initialization.")
            self.asr_model = None
            self.diarization_pipeline = None
            return

        logger.info(f"Initializing neural pipelines on {self.device.upper()}...")

        logger.info("Loading Faster-Whisper...")
        self.asr_model = WhisperModel(
            "large-v3", device=self.device, compute_type=self.compute_type
        )

        # Fetch the token securely from the .env file
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is missing. Check your .env file.")

        logger.info("Loading Pyannote directly from Hugging Face Hub...")
        try:
            self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")

            if self.device == "cuda":
                self.diarization_pipeline.to(torch.device("cuda"))

        except Exception as e:
            logger.error(f"Failed to load Pyannote from Hugging Face API: {e}")
            logger.warning(
                "Check your internet connection, HF token, and ensure you accepted the terms."
            )
            self.diarization_pipeline = None

        logger.info("STUB: Laughter Detector initialization skipped.")
        logger.info("STUB: Ollama connection skipped.")

    def separate_sources(self, audio_path: str, sketch_id: str) -> dict:
        out_dir = os.path.join(self.output_dir, sketch_id, "demucs_output")
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        stem_dir = os.path.join(out_dir, "htdemucs", base_name)

        # IF DRY RUN: Skip processing but return the paths that *would* be created
        if self.dry_run:
            logger.info(f"[DRY RUN] Skipping HTDemucs source separation for {sketch_id}.")
            return {
                "vocals": os.path.join(stem_dir, "vocals.wav"),
                "accompaniment": os.path.join(stem_dir, "no_vocals.wav"),
            }

        logger.info(f"Executing HTDemucs source separation for {sketch_id}...")
        cmd = f'--two-stems vocals -n htdemucs -j 8 --out "{out_dir}" "{audio_path}"'
        demucs.separate.main(shlex.split(cmd))

        return {
            "vocals": os.path.join(stem_dir, "vocals.wav"),
            "accompaniment": os.path.join(stem_dir, "no_vocals.wav"),
        }

    def detect_laughter(self, accompaniment_path: str, threshold=0.5, min_length=0.2) -> list:
        if self.dry_run:
            logger.info("[DRY RUN] Skipping biological punchline mapping.")
            return []

        logger.info("STUB: Skipping biological punchline mapping.")
        return []

    def transcribe_and_diarize(self, vocals_path: str) -> list:
        # IF DRY RUN: Return dummy script data
        if self.dry_run:
            logger.info("[DRY RUN] Skipping ASR and Diarization. Injecting dummy text.")
            return [
                {
                    "speaker": "SPEAKER_TEST",
                    "text": "Dry run verification successful.",
                    "start": 0.0,
                    "end": 2.0,
                }
            ]

        logger.info("Executing VAD-filtered transcription and speaker clustering...")

        segments, info = self.asr_model.transcribe(
            vocals_path, word_timestamps=True, vad_filter=True, language="pt"
        )

        words = []
        for segment in segments:
            for word in segment.words:
                words.append({"word": word.word, "start": word.start, "end": word.end})

        speaker_segments = []
        if self.diarization_pipeline:
            diarization = self.diarization_pipeline(vocals_path)
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speaker_segments.append({"speaker": speaker, "start": turn.start, "end": turn.end})
        else:
            speaker_segments = [{"speaker": "SPEAKER_00", "start": 0.0, "end": 9999.0}]

        aligned_script = []
        current_sentence = ""
        current_speaker = None
        sentence_start = 0.0

        for w in words:
            best_speaker = "UNKNOWN"
            max_overlap = 0

            for spk in speaker_segments:
                overlap = max(0, min(w["end"], spk["end"]) - max(w["start"], spk["start"]))
                if overlap > max_overlap:
                    max_overlap = overlap
                    best_speaker = spk["speaker"]

            if best_speaker != current_speaker:
                if current_sentence:
                    aligned_script.append(
                        {
                            "speaker": current_speaker,
                            "text": current_sentence.strip(),
                            "start": sentence_start,
                            "end": w["start"],
                        }
                    )
                current_speaker = best_speaker
                current_sentence = w["word"]
                sentence_start = w["start"]
            else:
                current_sentence += " " + w["word"]

        if current_sentence:
            aligned_script.append(
                {
                    "speaker": current_speaker,
                    "text": current_sentence.strip(),
                    "start": sentence_start,
                    "end": words[-1]["end"],
                }
            )

        return aligned_script

    def annotate_irony(self, aligned_script: list, laughs: list) -> list:
        if self.dry_run:
            logger.info("[DRY RUN] Skipping semantic LLM annotation.")
            status_message = "Dry Run Complete."
        else:
            logger.info("STUB: Skipping semantic LLM annotation.")
            status_message = "Stubbed. Awaiting LLM integration."

        for line in aligned_script:
            line["is_punchline"] = False
            line["semantic_metadata"] = {"status": status_message}

        return aligned_script

    def process_sketch_remote(self, sketch_id: str):
        audio_path_mp3 = os.path.join(self.output_dir, "{}.mp3".format(sketch_id))
        audio_path_wav = os.path.join(self.output_dir, "{}.wav".format(sketch_id))

        audio_path = audio_path_mp3 if os.path.exists(audio_path_mp3) else audio_path_wav

        if not os.path.exists(audio_path):
            logger.error(f"Audio file missing for {sketch_id}. Skipping.")
            return False

        mode_text = "[DRY RUN]" if self.dry_run else "GPU Processing"
        logger.info(f"Commencing {mode_text} for {sketch_id}...")

        try:
            stems = self.separate_sources(audio_path, sketch_id)
            laughs = self.detect_laughter(stems["accompaniment"])
            script = self.transcribe_and_diarize(stems["vocals"])
            final_corpus_entry = self.annotate_irony(script, laughs)

            out_file = os.path.join(self.output_json_dir, f"{sketch_id}_annotated.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(final_corpus_entry, f, ensure_ascii=False, indent=4)

            logger.info(f"Luso-Laugh corpus entry serialized to {out_file}")
            return True

        except Exception as e:
            logger.error(f"Pipeline crashed for {sketch_id}: {e}")
            return False


# === Execution Loop ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LusoLaugh Remote Orchestrator")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the pipeline logic without loading or executing ML models.",
    )
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

    generator = LusoLaughDatasetGenerator(
        dry_run=args.dry_run, output_dir=output_dir, output_json_dir=output_json_dir
    )

    for index, row in ready_sketches.iterrows():
        sketch_id = str(row["sketch_id"])
        logger.info(f"--- Processing Sketch ID: {sketch_id} ---")

        success = generator.process_sketch_remote(sketch_id)

        if success:
            if not args.dry_run:
                # Only save to the CSV if this is a real run
                df.at[index, "status"] = "processed"
                df.to_csv(catalog_file, index=False)
            else:
                logger.info(f"[DRY RUN] Status for {sketch_id} kept as 'downloaded'.")
