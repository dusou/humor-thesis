import argparse
import json
import logging
from pathlib import Path
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

SELECTED_SKETCHES = [
    "8gumwImdYwU",  # 242w | língua / media
    "vxF0uIY7t8Y",  # 227w | social / família
    "J7S9B85Js4I",  # 212w | media / absurdo
    "4LFRK7Rk3oU",  # 199w | defesa
    "L7Cu6ijCmMc",  # 179w | política
    "G0DJWvGgYOc",  # 170w | história / media
    "5fh4CQUpz5k",  # 148w | absurdo
    "Nk0YONoLEek",  # 122w | política
    "Ge7q7tNHRZs",  # 237w | política
    "9RMfvmcD12U",  # 176w | media / Eurovisão
    "fuvF8s-9xZU",  # 255w | Não Faleci Nada
]

LINE_RE = re.compile(r"^\[([^\]]+)\]:\s*(.+)$")


def split_output(output: str) -> tuple:
    if "</think>" in output:
        head, body = output.split("</think>", 1)
        return head.replace("<think>", "").strip(), body.strip()
    return "", output.strip()


def describe(script: str) -> dict:
    lines = [l.strip() for l in script.split("\n") if l.strip()]
    turns = [LINE_RE.match(l) for l in lines]
    cast = sorted({m.group(1).strip() for m in turns if m})
    return {
        "n_lines": len(lines),
        "n_turns": sum(1 for m in turns if m),
        "n_words": len(script.split()),
        "cast": cast,
        "n_characters": len(cast),
    }


def main(dataset_path: Path, gold_path: Path, prompts_path: Path, format_type: str) -> None:
    if not dataset_path.exists():
        logger.error(f"Dataset not found at {dataset_path}")
        return

    rows = {}
    for line in dataset_path.open(encoding="utf-8"):
        row = json.loads(line)
        rows.setdefault(row["sketch_id"], row)

    logger.info(f"Loaded {len(rows)} sketches from {dataset_path.name}")

    gold_records = []
    prompt_records = []
    missing = []

    for sketch_id in SELECTED_SKETCHES:
        row = rows.get(sketch_id)
        if row is None:
            missing.append(sketch_id)
            continue

        reasoning, script = split_output(row["output"])
        premise = row["input"].strip()
        stats = describe(script)

        gold_records.append(
            {
                "theme": sketch_id,
                "link": f"https://www.youtube.com/watch?v={sketch_id}",
                "format": format_type,
                "prompt": premise,
                "unedited_output": script,
                "edited_output": "[TODO]",
                **stats,
            }
        )

        prompt_records.append(
            {
                "theme": sketch_id,
                "format": format_type,
                "prompt": premise,
            }
        )

        logger.info(f"{sketch_id}: {stats['n_words']}w, {stats['n_turns']} turns, {stats['n_characters']} characters")

    if missing:
        logger.warning(f"Not found in dataset: {missing}")

    if not gold_records:
        logger.error("No sketches extracted; nothing written.")
        return

    gold_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.parent.mkdir(parents=True, exist_ok=True)

    gold_path.write_text(json.dumps(gold_records, ensure_ascii=False, indent=4), encoding="utf-8")
    prompts_path.write_text(json.dumps(prompt_records, ensure_ascii=False, indent=4), encoding="utf-8")

    words = [r["n_words"] for r in gold_records]
    turns = [r["n_turns"] for r in gold_records]
    logger.info(
        f"Extracted {len(gold_records)} sketches | words {min(words)}-{max(words)} | turns {min(turns)}-{max(turns)}"
    )
    logger.info(f"Gold standard: {gold_path}")
    logger.info(f"Prompt file:   {prompts_path}")


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Build the human-authored gold standard set")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=(script_dir / "../../data/04_lora_ready/lora_instruction_dataset.jsonl").resolve(),
    )
    parser.add_argument(
        "--gold-output",
        type=Path,
        default=(script_dir / "ComedIA_Human_outputs.json").resolve(),
    )
    parser.add_argument(
        "--prompts-output",
        type=Path,
        default=(script_dir / "eval_dataset_prompts.json").resolve(),
    )
    parser.add_argument("--format", type=str, default="sketch")
    args = parser.parse_args()

    main(args.dataset, args.gold_output, args.prompts_output, args.format)
