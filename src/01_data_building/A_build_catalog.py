import logging
import pandas as pd
from pathlib import Path
import re
import time
from typing import Dict, List
import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_targets(file_path: Path) -> List[Dict]:
    """Loads CSV targets into a list of dictionaries, returning an empty list if missing."""
    if not file_path.exists():
        logger.warning(f"Couldn't find '{file_path}'. Skipping.")
        return []

    try:
        targets = pd.read_csv(file_path).to_dict("records")
        logger.info(f"Loaded {len(targets)} target(s) from {file_path.name}.")
        return targets
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return []


def extract_metadata(targets: List[Dict], is_playlist: bool = True) -> pd.DataFrame:
    """
    Extracts video metadata using yt-dlp.
    Uses 'extract_flat' for fast channel scraping, or deep extraction for individual videos.
    """
    if not targets:
        return pd.DataFrame()

    logger.info(f"Extracting metadata for {len(targets)} target(s)...")
    all_videos = []

    ydl_opts = {
        "extract_flat": is_playlist,
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "http_headers": {"Accept-Language": "pt-PT,pt;q=0.9"},
        "extractor_args": {"youtube": {"lang": ["pt-PT", "pt"]}},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for target in targets:
            url = target.get("url")
            if not url or pd.isna(url):
                continue

            fallback_lang = target.get("fallback_language")
            manual_start = target.get("start_time")
            manual_end = target.get("end_time")

            logger.info(f"Processing: {url}")
            start_timer = time.time()

            try:
                info = ydl.extract_info(url, download=False)
                if not info:
                    continue

                # yt-dlp nests videos under "entries" for channels, but returns a flat dict for single videos
                entries = info.get("entries", [info]) if is_playlist else [info]

                for entry in entries:
                    if not entry:
                        continue

                    title = entry.get("title", "")

                    # skip deleted, private, or missing videos
                    if not title or title in ["[Deleted video]", "[Private video]"]:
                        continue

                    lang = entry.get("language") or fallback_lang
                    start_time = float(manual_start) if pd.notna(manual_start) else 0.0

                    if pd.notna(manual_end):
                        end_time = float(manual_end)
                    else:
                        duration = entry.get("duration")
                        end_time = float(duration) if duration is not None else None

                    channel_name = (
                        entry.get("uploader") or entry.get("uploader_id") or info.get("title")
                    )

                    all_videos.append(
                        {
                            "sketch_id": entry.get("id"),
                            "title": title,
                            "url": entry.get("webpage_url", entry.get("url", url)),
                            "start_time": start_time,
                            "end_time": end_time,
                            "declared_language": lang,
                            "channel_name": channel_name,
                            "status": "pending",
                        }
                    )

                logger.info(f"Done in {time.time() - start_timer:.2f} seconds.")

            except Exception as e:
                logger.error(f"Failed to process {url}. Reason: {e}")

    return pd.DataFrame(all_videos)


def clean_titles(df: pd.DataFrame) -> pd.DataFrame:
    """Strips the channel name and trailing/leading separators from video titles."""
    if df.empty:
        return df

    logger.info("Cleaning up sketch titles...")
    df_clean = df.copy()

    def sanitize(row):
        title = str(row["title"])
        channel = str(row["channel_name"])
        clean_title = re.sub(re.escape(channel), "", title, flags=re.IGNORECASE)
        clean_title = re.sub(r"^[\-\|:]\s*|[\-\|:]\s*$", "", clean_title).strip()

        return clean_title or title

    df_clean["title"] = df_clean.apply(sanitize, axis=1)
    return df_clean


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent.parent

    channels_csv = base_dir / "config" / "source_channels.csv"
    videos_csv = base_dir / "config" / "source_videos.csv"
    output_csv = base_dir / "data" / "01_catalogs" / "luso_laugh_catalog.csv"

    channel_targets = load_targets(channels_csv)
    video_targets = load_targets(videos_csv)

    df_channels = extract_metadata(channel_targets, is_playlist=True)
    df_videos = extract_metadata(video_targets, is_playlist=False)

    logger.info("Merging channel and manual video datasets...")
    combined_df = pd.concat([df_channels, df_videos], ignore_index=True)

    if not combined_df.empty:
        initial_count = len(combined_df)

        combined_df.dropna(subset=["url"], inplace=True)
        combined_df.drop_duplicates(subset=["sketch_id"], keep="last", inplace=True)

        dropped = initial_count - len(combined_df)
        if dropped > 0:
            logger.info(f"Cleaned up {dropped} invalid or duplicate entries.")

        final_df = combined_df  # clean_titles(combined_df)

        logger.info("\nPreview of final dataset:")
        print(final_df.head())

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        final_df.to_csv(output_csv, index=False)

        logger.info(
            f"Catalog generation complete. {len(final_df)} videos saved to {output_csv.name}"
        )
    else:
        logger.error("No data was extracted. Please check your input CSVs.")
