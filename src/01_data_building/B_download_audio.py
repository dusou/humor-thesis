import logging
import os
import pandas as pd
import yt_dlp
from yt_dlp.utils import download_range_func

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class LocalAudioIngestor:
    """
    Handles the bandwidth-intensive ingestion of audio payloads.
    """

    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def ingest_audio(self, url: str, sketch_id: str, start_sec: float = 0.0, end_sec: float = None) -> str:
        logger.info(f"Commencing automated ingestion for {sketch_id}...")
        save_path = os.path.join(self.output_dir, "{}.mp3".format(sketch_id))

        # Configure yt-dlp to extract the highest fidelity uncompressed audio
        ydl_opts = {
            "outtmpl": os.path.join(self.output_dir, sketch_id),
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

        # Apply excerpt boundaries if an end time exists
        if end_sec is not None and end_sec > start_sec:
            logger.info(f"Extracting specific segment: {start_sec}s to {end_sec}s")
            ydl_opts["download_ranges"] = download_range_func(None, [(start_sec, end_sec)])
            ydl_opts["force_keyframes_at_cuts"] = True

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])
            return save_path
        except Exception as e:
            logger.error(f"Ingestion failed for {sketch_id}: {e}")
            return None


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    catalog_file = os.path.join(dirname, "../../data/01_catalogs/luso_laugh_catalog.csv")

    if not os.path.exists(catalog_file):
        logger.error(f"Catalog file '{catalog_file}' not found. Please run the metadata extraction script first.")
        exit()

    df = pd.read_csv(catalog_file)

    # Filter for items that are marked as 'pending'
    pending_sketches = df[df["status"] == "pending"]
    logger.info(f"Found {len(pending_sketches)} pending sketches to download.")

    output_dir = os.path.normpath(os.path.join(dirname, "../../data/02_audio_corpus/"))
    ingestor = LocalAudioIngestor(output_dir=output_dir)

    for index, row in pending_sketches.iterrows():
        sketch_id = str(row["sketch_id"])
        url = str(row["url"])

        # Safely handle float conversions, accounting for pandas NaN values
        start_sec = float(row["start_time"]) if pd.notna(row["start_time"]) else 0.0
        end_sec = float(row["end_time"]) if pd.notna(row["end_time"]) else None

        logger.info(f"--- Downloading Sketch: {row['title']} ({sketch_id}) ---")

        # Trigger the yt-dlp download
        audio_path = ingestor.ingest_audio(url, sketch_id, start_sec, end_sec)

        # Verify if the file was created and update the CSV
        if audio_path and os.path.exists(audio_path):
            df.at[index, "status"] = "downloaded"  # Change status to mark it ready for the cluster
            logger.info(f"Successfully saved and updated status for {sketch_id}.")
        else:
            logger.error(f"Failed to verify downloaded file for {sketch_id}. Status remains 'pending'.")

    df.to_csv(catalog_file, index=False)  # Save progress immediately

    logger.info("Download phase complete.")
