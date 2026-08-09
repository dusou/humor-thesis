import logging
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm
from typing import Optional
import yt_dlp
from yt_dlp.utils import download_range_func

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class LocalAudioIngestor:
    """
    Handles the ingestion of audio payloads.
    """

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def ingest_audio(
        self,
        url: str,
        sketch_id: str,
        start_sec: float = 0.0,
        end_sec: Optional[float] = None,
        max_retries: int = 3,
    ) -> Optional[Path]:

        save_path = self.output_dir / f"{sketch_id}.mp3"
        out_template = str(self.output_dir / sketch_id)

        ydl_opts = {
            "outtmpl": out_template,
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "256",
                }
            ],
            "quiet": True,
            "no_warnings": True,
        }

        # Apply excerpt boundaries
        if end_sec is not None and end_sec > start_sec:
            ydl_opts["download_ranges"] = download_range_func(None, [(start_sec, end_sec)])
            ydl_opts["force_keyframes_at_cuts"] = True

        # Exponential backoff retry loop
        for attempt in range(1, max_retries + 1):
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])

                if save_path.exists():
                    return save_path

            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"Ingestion failed for {sketch_id} after {max_retries} attempts: {e}"
                    )
                    return None

                # Calculate delay: 5s, 10s, 20s...
                delay = 5 * (2 ** (attempt - 1))
                tqdm.write(
                    f"Warning: Attempt {attempt} failed for {sketch_id}. Retrying in {delay}s..."
                )
                time.sleep(delay)

        return None


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    base_dir = script_dir.parent.parent

    catalog_file = base_dir / "data" / "01_catalogs" / "luso_laugh_catalog.csv"
    output_dir = base_dir / "data" / "02_audio_corpus"

    if not catalog_file.exists():
        logger.error(
            f"Catalog file '{catalog_file.name}' not found. Please run metadata extraction first."
        )
        exit(1)

    df = pd.read_csv(catalog_file)

    # Filter for items that are marked as 'pending'
    pending_sketches = df[df["status"] == "pending"]

    if pending_sketches.empty:
        logger.info("No pending sketches found. Your corpus is up to date!")
        exit(0)

    logger.info(f"Found {len(pending_sketches)} pending sketch(es). Starting ingestion...")
    ingestor = LocalAudioIngestor(output_dir=output_dir)

    progress_bar = tqdm(
        pending_sketches.iterrows(),
        total=len(pending_sketches),
        desc="Downloading Corpus",
        unit="sketch",
        dynamic_ncols=True,
    )

    for index, row in progress_bar:
        sketch_id = str(row["sketch_id"])
        url = str(row["url"])
        title = str(row["title"])

        progress_bar.set_postfix({"Current": title[:25] + ("..." if len(title) > 25 else "")})

        start_sec = float(row["start_time"]) if pd.notna(row["start_time"]) else 0.0
        end_sec = float(row["end_time"]) if pd.notna(row["end_time"]) else None

        audio_path = ingestor.ingest_audio(url, sketch_id, start_sec, end_sec)

        # Verify if the file was created and update the CSV status
        if audio_path and audio_path.exists():
            df.at[index, "status"] = "downloaded"
            df.to_csv(catalog_file, index=False)
        else:
            tqdm.write(f"❌ Failed to download: {sketch_id}")

        # Small buffer to be polite to the YouTube API between successful requests
        time.sleep(2)

    logger.info("Audio ingestion phase complete!")
