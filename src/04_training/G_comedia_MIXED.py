import argparse
import gc
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os
from pathlib import Path
from peft import PeftModel
import re
import sys
import torch
from tqdm import tqdm
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
    StoppingCriteria,
    StoppingCriteriaList,
)
from typing import Dict, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

transformers.logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

data_process_dir = (Path(__file__).resolve().parent.parent / "02_data_process").resolve()
if str(data_process_dir) not in sys.path:
    sys.path.insert(0, str(data_process_dir))

from D_data_transform import MACRO_INSTRUCTION, SYSTEM_PROMPT


class ThinkCloseStoppingCriteria(StoppingCriteria):
    """
    Thinking Block stopping criteria
    """

    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len

    def __call__(self, input_ids, scores, **kwargs):
        tail_ids = input_ids[0, self.prompt_len :]
        tail_text = self.tokenizer.decode(tail_ids[-8:], skip_special_tokens=False)
        return "</think>" in tail_text


class RepetitionControlProcessor(LogitsProcessor):
    """Penalise repetition within a recent window only."""

    def __init__(self, presence, frequency, prompt_len, max_freq=4.0, window=64):
        self.presence, self.frequency = presence, frequency
        self.prompt_len, self.max_freq, self.window = prompt_len, max_freq, window

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.prompt_len :]
        for i in range(scores.shape[0]):
            recent = gen[i][-self.window :]
            if not recent.numel():
                continue
            toks, counts = torch.unique(recent, return_counts=True)
            scores[i, toks] -= self.presence + torch.clamp(self.frequency * counts.to(scores.dtype), max=self.max_freq)
        return scores


class ComediaHybridGenerator:
    """
    Implements Architecture C (Hybrid Approach) leveraging both
    Retrieval-Augmented Generation and LoRA Fine-Tuning.
    """

    def __init__(
        self,
        adapter_path: Path,
        corpus_dir: str = "../../data/04_rag_ready",
        base_model_name: str = "Qwen/Qwen3.5-9B",
        embedder_model_name: str = "BAAI/bge-m3",
        dry_run: bool = False,
    ) -> None:
        self.corpus_dir = Path(corpus_dir)
        self.dry_run = dry_run
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.hf_token = os.getenv("HF_TOKEN")
        self.vector_store = None
        self.tokenizer = None
        self.model = None

        self._build_vector_store(embedder_model_name)
        self._initialize_model(base_model_name, adapter_path)

    def _build_vector_store(self, embedder_model_name: str) -> None:
        """Loads and ingests the Luso-Laugh corpus into an ephemeral ChromaDB instance."""
        if not self.corpus_dir.exists():
            logger.error(f"RAG corpus directory '{self.corpus_dir}' not found.")
            return

        json_files = list(self.corpus_dir.glob("*_rag.json"))
        logger.info(f"Loading {len(json_files)} documents from the corpus...")

        lc_documents = []
        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)
                    search_text = f"{doc_data.get('content', '')}\n{doc_data.get('comedic_metadata', '')}"
                    metadata = {
                        "sketch_id": doc_data.get("sketch_id", "UNKNOWN"),
                        "clean_content": doc_data.get("content", ""),
                    }
                    lc_documents.append(Document(page_content=search_text, metadata=metadata))
            except Exception as e:
                logger.warning(f"Failed to load {filepath.name}: {e}")

        logger.info(f"Initializing Embeddings ({embedder_model_name})...")
        gpu_embeddings = HuggingFaceEmbeddings(
            model_name=embedder_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Ingesting documents into ephemeral ChromaDB...")
        self.vector_store = Chroma.from_documents(
            documents=lc_documents, embedding=gpu_embeddings, collection_metadata={"hnsw:space": "cosine"}
        )

        del gpu_embeddings
        gc.collect()
        torch.cuda.empty_cache()

        logger.info("Swapping to CPU embeddings for query-time retrieval...")
        self.vector_store._embedding_function = HuggingFaceEmbeddings(
            model_name=embedder_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    def _initialize_model(self, base_model_name: str, adapter_path: Path) -> None:
        """Loads the Base LLM in 16-bit precision and dynamically merges the LoRA adapter."""
        if self.dry_run:
            logger.info("DRY RUN: Skipping LLM initialization.")
            return

        if not adapter_path.exists():
            logger.error(f"Adapter not found at {adapter_path}. Please train it first.")
            return

        logger.info(f"Loading Base LLM: {base_model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, token=self.hf_token)

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                dtype=self.compute_type,
                device_map="auto",
                token=self.hf_token,
                attn_implementation="sdpa",
            )

            logger.info(f"Merging LoRA Adapter: {adapter_path.name}")
            self.model = PeftModel.from_pretrained(base_model, str(adapter_path))
            self.model.eval()
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            self.model = None

    @staticmethod
    def _normalise_reasoning(reasoning_text: str, trim_incomplete: bool = True) -> str:
        closed = "</think>" in reasoning_text

        body = reasoning_text.split("</think>")[0]
        body = re.sub(r"</?think>", "", body).strip()

        if not closed and trim_incomplete:
            cut = max(body.rfind(". "), body.rfind(".\n"), body.rfind("! "), body.rfind("? "))
            if cut > 200:
                body = body[: cut + 1]

        return f"<think>\n{body}\n</think>\n\n"

    def _generate_bounded(
        self, messages: list, reasoning_budget: int = 5000, answer_budget: int = 2250
    ) -> Tuple[str, str]:
        """Capped reasoning block generation followed by a guaranteed answer generation."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                prompt_len = inputs["input_ids"].shape[1]
                out_reasoning = self.model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    min_p=0.0,
                    repetition_penalty=1.0,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.05, 0.3, prompt_len)]),
                    stopping_criteria=StoppingCriteriaList([ThinkCloseStoppingCriteria(self.tokenizer, prompt_len)]),
                )
                reasoning_text = self.tokenizer.decode(
                    out_reasoning[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False
                )
        finally:
            del inputs, out_reasoning
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if "</think>" not in reasoning_text:
            logger.warning(f"Reasoning did not close within {reasoning_budget} tokens.")
        reasoning_text = self._normalise_reasoning(reasoning_text)

        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)

        try:
            with torch.inference_mode():
                prompt_len = inputs_answer["input_ids"].shape[1]
                out_answer = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=answer_budget,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.05, 0.3, prompt_len)]),
                )
                answer_text = self.tokenizer.decode(
                    out_answer[0, inputs_answer["input_ids"].shape[1] :], skip_special_tokens=True
                )
        finally:
            del inputs_answer, out_answer
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        clean_answer = re.sub(r"<think>[\s\S]*?</think>", "", answer_text)
        clean_answer = re.sub(r"</?think>", "", clean_answer)
        clean_answer = re.sub(r"^(?:assistant\s*)+", "", clean_answer.strip(), flags=re.IGNORECASE)

        clean_reasoning = re.sub(r"</?think>", "", reasoning_text).strip()
        return clean_reasoning, clean_answer.strip()

    def generate(self, query: str, format_type: str = "sketch") -> Dict:
        """Executes the Hybrid pipeline: Context retrieval + LoRA stylized generation."""
        if self.dry_run or not self.model:
            return {"text": f"[DRY RUN] Generated mock {format_type} output.", "sources": []}

        try:
            # RAG Retrieval Layer
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=2)
            SCORE_THRESHOLD = 0.50
            filtered = [(doc, score) for doc, score in docs_with_scores if score <= SCORE_THRESHOLD]

            retrieved_contexts = []
            for doc, score in filtered:
                logger.info(
                    f'\tscore={score:.4f} | id={doc.metadata["sketch_id"]} | preview="{doc.metadata["clean_content"][:30]}"'
                )
                s_id = doc.metadata.get("sketch_id", "UNKNOWN")
                s_text = doc.metadata.get("clean_content", "")
                retrieved_contexts.append({"sketch_id": s_id, "text": s_text})

            docs = [doc for doc, _ in filtered]
            formatted_context = ""
            for i, doc in enumerate(docs, 1):
                if len(formatted_context) + len(doc.metadata["sketch_id"]) > 15000:
                    logger.info(f"Maximum context size reached. Using only {i - 1} documents.")
                    break
                formatted_context += (
                    f"--- EXEMPLO {i} ({doc.metadata['sketch_id']}) ---\n{doc.metadata['clean_content']}\n"
                )

            if filtered:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            "Aqui estão sketches de referência para inspiração de ritmo, "
                            f"cadência e registo:\n\n{formatted_context}\n\n"
                            f"{MACRO_INSTRUCTION}\n\n{query}"
                        ),
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"{MACRO_INSTRUCTION}\n\n{query}"},
                ]
                logger.warning(f"No sufficiently relevant sketches found for query: {query[:60]}...")

            # LoRA Generation Layer
            reasoning, response = self._generate_bounded(messages)
            return {
                "response": response,
                "sources": retrieved_contexts,
                "reasoning": reasoning,
            }

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return {
                "response": "ERROR: Generation failed.",
                "sources": [],
                "reasoning": "ERROR: Generation failed.",
            }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComedIA Hybrid (RAG + LoRA) Generator")
    parser.add_argument(
        "--rank",
        type=int,
        default=16,
        choices=[4, 8, 16, 32, 64],
        help="LoRA rank dimension to load the respective adapter.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Run without loading Qwen model")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    corpus_dir = (script_dir / "../../data/04_rag_ready").resolve()
    input_path = (script_dir / "input_prompts.json").resolve()

    # Differentiates output based on the chosen rank
    output_path = (script_dir / f"../../data/06_comedia_outputs/ComedIA_Hybrid_r{args.rank}_outputs.json").resolve()
    adapter_path = (script_dir / "adapters" / f"comedia_lora_r{args.rank}").resolve()

    if not input_path.exists():
        logger.error(f"Input file '{input_path}' not found.")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)

    logger.info(f"Loaded {len(prompts_list)} prompt pairs.")

    hybrid_system = ComediaHybridGenerator(adapter_path=adapter_path, corpus_dir=str(corpus_dir), dry_run=args.dry_run)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = []

    set_seed(42)

    for item in tqdm(prompts_list, desc=f"Generating Hybrid (r={args.rank})", unit="prompt"):
        theme = item.get("theme", "general")
        format_type = item.get("format", "sketch")
        prompt = item.get("prompt", "")

        generation_result = hybrid_system.generate(query=prompt, format_type=format_type)

        output_data.append(
            {
                "theme": theme,
                "format": format_type,
                "prompt": prompt,
                "retrieved_context": generation_result["sources"],
                "reasoning": generation_result.get("reasoning", ""),
                "output": generation_result.get("response", ""),
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Hybrid generation complete. Results saved to {output_path.name}")
