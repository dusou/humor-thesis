import argparse
import json
import logging
import os
from pathlib import Path
from tqdm import tqdm
import transformers
from transformers import AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"


class DatasetTokenAnalyzer:
    """
    Analyzes the token distribution of the formatted LoRA dataset
    to validate context window constraints prior to fine-tuning.
    """

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens
        self.model_name = "Qwen/Qwen3.5-9B"
        self.hf_token = os.getenv("HF_TOKEN")
        self.tokenizer = None

        script_dir = Path(__file__).resolve().parent
        self.dataset_path = (
            script_dir.parent.parent / "data" / "04_lora_ready" / "lora_instruction_dataset.jsonl"
        ).resolve()

    def _init_tokenizer(self) -> None:
        logger.info(f"Initializing tokenizer: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=self.hf_token)

    def analyze(self) -> None:
        if not self.dataset_path.exists():
            logger.error(f"Dataset not found at: {self.dataset_path}")
            return

        if not self.tokenizer:
            self._init_tokenizer()

        total_tokens = 0
        highest_token_count = 0
        total_examples = 0
        outliers_count = 0

        logger.info(f"Scanning dataset against {self.max_tokens} token limit...")

        with open(self.dataset_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in tqdm(lines, desc="Analyzing tokens", unit="sketch"):
            data = json.loads(line)

            # Reconstruct the expected ChatML format
            sys_msg = "És um argumentista profissional de comédia e sátira portuguesa."
            user_msg = f"{data.get('instruction', '')}\n\n{data.get('input', '')}"
            assistant_msg = data.get("output", "")

            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]

            formatted_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
            tokenized_output = self.tokenizer(formatted_prompt)
            num_tokens = len(tokenized_output["input_ids"])

            highest_token_count = max(highest_token_count, num_tokens)
            total_tokens += num_tokens
            total_examples += 1

            if num_tokens > self.max_tokens:
                outliers_count += 1

        if total_examples == 0:
            logger.warning("Dataset is empty.")
            return

        avg_tokens = total_tokens / total_examples
        outlier_percentage = (outliers_count / total_examples) * 100

        logger.info("=" * 45)
        logger.info(f"TOKEN LENGTH DIAGNOSTICS ({self.max_tokens} LIMIT)")
        logger.info("=" * 45)
        logger.info(f"Total examples:         {total_examples}")
        logger.info(f"Average length:         {avg_tokens:.0f} tokens")
        logger.info(f"Maximum length:         {highest_token_count} tokens")
        logger.info(f"Over limit (> {self.max_tokens}):   {outliers_count} ({outlier_percentage:.1f}%)")
        logger.info("=" * 45)

        if outliers_count == 0:
            logger.info("Status: Optimal. Zero data truncation expected.")
        elif outlier_percentage < 5.0:
            logger.info("Status: Acceptable. Minimal truncation (<5%). Ready for training.")
        else:
            logger.warning("Status: Suboptimal. High outlier count. Consider filtering the dataset.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze token lengths of the LoRA dataset.")
    parser.add_argument("--limit", type=int, default=4096, help="Maximum token limit to check against (default: 4096).")
    args = parser.parse_args()

    analyzer = DatasetTokenAnalyzer(max_tokens=args.limit)
    analyzer.analyze()
