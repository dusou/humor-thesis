import logging
import os
import pandas as pd
import re
import time
import yt_dlp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_targets_from_csv(file_path: str) -> list:
    """Reads target channels and fallback languages from an external CSV."""
    if not os.path.exists(file_path):
        logger.error(f"Target file '{file_path}' not found. Please create it.")
        return []

    try:
        targets_df = pd.read_csv(file_path)
        targets_list = targets_df.to_dict("records")
        logger.info(f"Successfully loaded {len(targets_list)} target(s) from {file_path}.")
        return targets_list
    except Exception as e:
        logger.error(f"Failed to parse {file_path}. Error: {e}")
        return []


def build_channel_catalog(targets: list) -> pd.DataFrame:
    """
    Extracts video metadata from a list of targets.
    """
    if not targets:
        logger.warning("No targets provided. Exiting extraction.")
        return pd.DataFrame()

    logger.info(f"Initiating metadata extraction for {len(targets)} channel(s).")

    all_videos = []

    # Configure yt-dlp to only extract metadata, not download media
    ydl_opts = {"extract_flat": True, "quiet": True, "no_warnings": True, "ignoreerrors": True}

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for target in targets:
            url = target.get("url")
            fallback_lang = target.get("fallback_language")

            if not url or pd.isna(url):
                continue

            logger.info(f"Connecting to channel: {url}")
            start_time = time.time()

            try:
                # Extract channel data
                channel_info = ydl.extract_info(url, download=False)

                if "entries" in channel_info:
                    entries_count = 0
                    for entry in channel_info["entries"]:
                        if entry:
                            # Use YouTube's declared language if available, otherwise use the CSV fallback
                            extracted_lang = entry.get("language")
                            final_lang = extracted_lang if extracted_lang else fallback_lang

                            # Safely extract duration and set temporal boundaries
                            duration = entry.get("duration")
                            end_time = float(duration) if duration is not None else None

                            video_data = {
                                "sketch_id": entry.get("id"),
                                "title": entry.get("title"),
                                "url": entry.get("url"),
                                "start_time": 0.0,
                                "end_time": end_time,
                                "declared_language": final_lang,
                                "channel_name": channel_info.get(
                                    "uploader", channel_info.get("title")
                                ),
                                "status": "pending",
                            }
                            all_videos.append(video_data)
                            entries_count += 1

                elapsed = time.time() - start_time
                logger.info(
                    f"Successfully mapped {entries_count} videos from channel in {elapsed:.2f} seconds."
                )

            except Exception as e:
                logger.error(f"Failed to process {url}. Reason: {e}")

    df = pd.DataFrame(all_videos)

    # Clean up any potential missing URLs or duplicates
    if not df.empty:
        initial_count = len(df)
        df = df.dropna(subset=["url"])
        df = df.drop_duplicates(subset=["sketch_id"])
        dropped_count = initial_count - len(df)

        if dropped_count > 0:
            logger.warning(f"Removed {dropped_count} invalid or duplicate entries during cleanup.")

    return df


def clean_sketch_titles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes the channel name from the 'title' column of the dataframe
    and cleans up any trailing/leading separators (-, |, :).
    """
    if df.empty:
        return df

    logger.info("Initializing sketch title sanitation process...")

    df_clean = df.copy()

    def strip_name(row):
        title = str(row["title"])
        channel = str(row["channel_name"])

        # Remove the channel name (case-insensitive)
        pattern = re.compile(re.escape(channel), re.IGNORECASE)
        cleaned_title = pattern.sub("", title)

        # Clean up leftover separators and whitespace at the end or beginning
        cleaned_title = re.sub(r"[\-\|:]\s*$", "", cleaned_title).strip()
        cleaned_title = re.sub(r"^[\-\|:]\s*", "", cleaned_title).strip()

        return cleaned_title if cleaned_title else title

    # Apply the logic row by row
    df_clean["title"] = df_clean.apply(strip_name, axis=1)

    logger.info("Title sanitation completed successfully.")
    return df_clean


def clean_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = clean_sketch_titles(df)
    return df_clean


if __name__ == "__main__":
    dirname = os.path.dirname(__file__)
    config_file = os.path.normpath(os.path.join(dirname, "../../config/source_channels.csv"))
    output_csv = os.path.normpath(
        os.path.join(dirname, "../../data/01_catalogs/luso_laugh_catalog.csv")
    )

    targets = load_targets_from_csv(config_file)

    # Extract metadata
    if targets:
        source_df = build_channel_catalog(targets)

        # Clean the metadata
        clean_df = clean_catalogue(source_df)

        # Preview the data
        logger.info("Previewing first 5 rows of the cleaned dataset:")
        print(clean_df.head())

        # Save to CSV to feed the main Luso-Laugh pipeline
        clean_df.to_csv(output_csv, index=False)
        logger.info(f"Catalog generation complete. {len(clean_df)} total videos extracted.")
        logger.info(f"Data successfully saved to: {output_csv}")
