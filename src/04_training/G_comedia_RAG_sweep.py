import argparse
import gc
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
import logging
import os
from pathlib import Path
import re
import sys
import torch
import transformers
from transformers import (
    LogitsProcessor,
    LogitsProcessorList,
    set_seed,
    StoppingCriteria,
    StoppingCriteriaList,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Reduce transformer warnings
transformers.logging.set_verbosity_error()
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

data_process_dir = (Path(__file__).resolve().parent.parent / "02_data_process").resolve()
if str(data_process_dir) not in sys.path:
    sys.path.insert(0, str(data_process_dir))

from D_data_transform import MACRO_INSTRUCTION, SYSTEM_PROMPT

K_VALUES = [1, 2, 3, 4]
K_SWEEP_THRESHOLD = 0.50

THRESHOLD_VALUES = [0.50, 0.48, 0.46, 0.44, 0.42, 0.40]
THRESHOLD_SWEEP_K = 4

SEED = 42


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


class ComediaRAG:
    """
    Implements Architecture A (RAG) using LangChain, ChromaDB, and local Qwen-3.5-9B.
    Retrieval parameters are passed per call so one loaded model serves the whole sweep.
    """

    def __init__(
        self,
        corpus_dir: str = "../../data/04_rag_ready",
        llm_model_name: str = "Qwen/Qwen3.5-9B",
        embedder_model_name: str = "BAAI/bge-m3",
        dry_run: bool = False,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.dry_run = dry_run

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.hf_token = os.getenv("HF_TOKEN")

        self.vector_store = None

        self._build_vector_store(embedder_model_name)
        self._init_chain(llm_model_name)

    def _build_vector_store(self, embedder_model_name: str):
        """Loads JSON files, wraps them in LangChain Documents, and ingests them into ChromaDB."""
        if not self.corpus_dir.exists():
            logger.error(f"RAG corpus directory '{self.corpus_dir}' not found.")
            return

        json_files = list(self.corpus_dir.glob("*_rag.json"))
        logger.info(f"Loading {len(json_files)} documents from the Luso-Laugh corpus into LangChain...")

        lc_documents = []
        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)

                    search_text = doc_data.get("content", "")
                    metadata = {
                        "sketch_id": doc_data.get("sketch_id", "UNKNOWN"),
                        "clean_content": doc_data.get("content", ""),
                    }

                    lc_documents.append(Document(page_content=search_text, metadata=metadata))
            except Exception as e:
                logger.warning(f"Failed to load {filepath.name}: {e}")

        logger.info(f"Initializing HuggingFace Embeddings ({embedder_model_name})...")
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

        # swap in a lightweight CPU embedder for query-time use during generation
        logger.info("Swapping to CPU embeddings for query-time retrieval...")
        self.vector_store._embedding_function = HuggingFaceEmbeddings(
            model_name=embedder_model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

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

    def _init_chain(self, llm_model_name: str):
        """Initializes Qwen-3.5-9B and constructs a decoupled LCEL chain."""
        if self.dry_run:
            logger.info("DRY RUN: Skipping Qwen LLM initialization.")
            return

        logger.info(f"Loading Generative LLM: {llm_model_name}...")
        try:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(llm_model_name, token=self.hf_token)
            self.model = transformers.AutoModelForCausalLM.from_pretrained(
                llm_model_name,
                dtype=self.compute_type,
                device_map="auto",
                token=self.hf_token,
                attn_implementation="sdpa",
            )
        except Exception as e:
            logger.error(f"Failed to load Generative LLM: {e}")
            self.model = None

    def _generate_bounded(self, messages, reasoning_budget=5000, answer_budget=2250):
        """Two bounded phases: capped reasoning, then a guaranteed answer budget."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:  # Phase 1 : reason
            with torch.inference_mode():
                prompt_len = inputs["input_ids"].shape[1]
                out1 = self.model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.95,
                    top_k=50,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.05, 0.3, prompt_len)]),
                    stopping_criteria=StoppingCriteriaList([ThinkCloseStoppingCriteria(self.tokenizer, prompt_len)]),
                )
            reasoning_text = self.tokenizer.decode(out1[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        finally:  # Memory Cleanup
            del inputs, out1
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if "</think>" not in reasoning_text:
            logger.warning(f"Reasoning did not close within {reasoning_budget} tokens.")
        reasoning_text = self._normalise_reasoning(reasoning_text)

        inputs_answer = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)
        try:  # Phase 2: answer
            with torch.inference_mode():
                prompt_len = inputs_answer["input_ids"].shape[1]
                out2 = self.model.generate(
                    **inputs_answer,
                    max_new_tokens=answer_budget,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    top_k=50,
                    logits_processor=LogitsProcessorList([RepetitionControlProcessor(1.05, 0.3, prompt_len)]),
                )
            answer_text = self.tokenizer.decode(
                out2[0, inputs_answer["input_ids"].shape[1] :], skip_special_tokens=True
            )
        finally:
            del inputs_answer
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        clean_answer = re.sub(r"<think>[\s\S]*?</think>", "", answer_text)
        clean_answer = re.sub(r"</?think>", "", clean_answer)
        clean_answer = re.sub(r"^(?:assistant\s*)+", "", clean_answer.strip(), flags=re.IGNORECASE)

        return reasoning_text, clean_answer.strip()

    def generate(self, query: str, k: int, threshold: float, format_type: str = "sketch") -> dict:
        if self.dry_run:
            return {
                "sources": [],
                "reasoning": f"[DRY RUN] k={k} threshold={threshold}",
                "response": f"[DRY RUN] Generated mock {format_type} output.",
            }

        try:
            # explicitly retrieve the documents
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=k)
            filtered = [(doc, score) for doc, score in docs_with_scores if score <= threshold]

            # extract IDs and actual text
            retrieved_contexts = []
            for doc, score in filtered:
                logger.info(
                    f'\tscore={score:.4f} // id={doc.metadata["sketch_id"]} // preview="{doc.metadata["clean_content"][:30]}"'
                )

                s_id = doc.metadata.get("sketch_id", "UNKNOWN")
                s_text = doc.metadata.get("clean_content", "")

                retrieved_contexts.append({"sketch_id": s_id, "score": score, "text": s_text})

            docs = [doc for doc, _ in filtered]

            # format the documents into a string context for the LLM Prompt
            formatted_context = ""
            for i, doc in enumerate(docs, 1):
                if len(formatted_context) + len(doc.metadata["clean_content"]) > 15000:
                    logger.info(f"Maximum context size reached. Using only {i - 1} documents.")
                    break
                formatted_context += f"--- EXEMPLO {i} ---\n{doc.metadata['clean_content']}\n"

            if formatted_context:
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

            reasoning, response = self._generate_bounded(messages)

            return {
                "sources": retrieved_contexts,
                "reasoning": reasoning,
                "response": response,
            }

        except Exception:
            logger.exception(f"Generation chain failed for prompt: {query[:30]}...")
            return {
                "sources": [],
                "reasoning": "ERROR: Generation failed.",
                "response": "ERROR: Generation failed.",
            }


def run_configuration(rag_system, prompts_list, k, threshold, output_path, overwrite):
    if output_path.exists() and not overwrite:
        logger.info(f"{output_path.name} already exists, skipping.")
        return

    logger.info(f"=== k={k} // threshold={threshold:.2f} // {output_path.name} ===")
    set_seed(SEED)

    output_data = []
    total_items = len(prompts_list)

    for idx, item in enumerate(prompts_list, start=1):
        theme = item.get("theme", "general")
        format_type = item.get("format", "sketch")
        prompt = item.get("prompt", "")

        logger.info(f"Processing ({idx}/{total_items}) // Theme: '{theme}' // Format: '{format_type}'")

        generation_result = rag_system.generate(query=prompt, k=k, threshold=threshold, format_type=format_type)

        output_data.append(
            {
                "theme": theme,
                "format": format_type,
                "prompt": prompt,
                "k": k,
                "score_threshold": threshold,
                "n_retrieved": len(generation_result["sources"]),
                "retrieved_context": generation_result["sources"],
                "reasoning": generation_result.get("reasoning", ""),
                "output": generation_result["response"],
            }
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    retrieved = [x["n_retrieved"] for x in output_data]
    logger.info(
        f"Saved {output_path.name} // documents retrieved: min {min(retrieved)} "
        f"max {max(retrieved)} mean {sum(retrieved) / len(retrieved):.1f}"
    )


def preview_retrieval(rag_system, prompts_list):
    logger.info("=== Retrieval preview ===")
    all_scores = []

    for item in prompts_list:
        hits = rag_system.vector_store.similarity_search_with_score(item.get("prompt", ""), k=max(K_VALUES))
        scores = [round(s, 3) for _, s in hits]
        all_scores += [s for _, s in hits]
        logger.info(f"{item.get('theme', '?')[:28]:30s} {scores}")

    all_scores.sort()
    logger.info(f"min {all_scores[0]:.3f} // median {all_scores[len(all_scores) // 2]:.3f} // max {all_scores[-1]:.3f}")
    for t in THRESHOLD_VALUES:
        kept = sum(1 for s in all_scores if s <= t)
        logger.info(f"\tthreshold {t:.2f} keeps {kept}/{len(all_scores)} documents")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComedIA RAG Parameter Sweep")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading Qwen model")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate existing configurations")
    parser.add_argument("--preview", action="store_true", help="Print retrieval scores and exit")
    parser.add_argument("--sweep", choices=["k", "threshold", "both"], default="both")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    corpus_dir = (script_dir / "../../data/04_rag_ready").resolve()
    input_path = (script_dir / "input_prompts.json").resolve()
    output_dir = (script_dir / "../../data/06_comedia_outputs").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # load input dataset
    if not input_path.exists():
        logger.error(f"Input file '{input_path}' not found.")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)

    logger.info(f"Loaded {len(prompts_list)} prompt pairs from {input_path}")

    # initialize RAG system
    rag_system = ComediaRAG(
        corpus_dir=str(corpus_dir),
        llm_model_name="Qwen/Qwen3.5-9B",
        dry_run=args.dry_run or args.preview,
    )

    if args.preview:
        preview_retrieval(rag_system, prompts_list)
        exit(0)

    configurations = []
    if args.sweep in ("k", "both"):
        for k in K_VALUES:
            configurations.append((k, K_SWEEP_THRESHOLD, output_dir / f"ComedIA_outputs_k{k}.json"))
    if args.sweep in ("threshold", "both"):
        for threshold in THRESHOLD_VALUES:
            tag = f"s{int(round(threshold * 100))}"
            configurations.append((THRESHOLD_SWEEP_K, threshold, output_dir / f"ComedIA_outputs_{tag}.json"))

    logger.info(f"Running {len(configurations)} configurations over {len(prompts_list)} prompts each")

    for k, threshold, path in configurations:
        run_configuration(rag_system, prompts_list, k, threshold, path, args.overwrite)

    logger.info("Sweep complete.")
