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
import torch
import transformers

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


class ComediaRAG:
    """
    Implements Architecture A (RAG) using LangChain, ChromaDB, and local Qwen-3.5-9B.
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

                    search_text = f"{doc_data.get('content', '')}\n{doc_data.get('comedic_metadata', '')}"
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

    def _generate_bounded(self, messages, reasoning_budget=4000, answer_budget=3000):
        """Two bounded phases: capped reasoning, then a guaranteed answer budget."""
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        try:  # Phase 1 : reason
            with torch.inference_mode():
                out1 = self.model.generate(
                    **inputs,
                    max_new_tokens=reasoning_budget,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    top_p=0.9,
                )
            reasoning_text = self.tokenizer.decode(out1[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)
        finally:  # Memory Cleanup
            del inputs
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        if "</think>" in reasoning_text:
            reasoning_text = reasoning_text.split("</think>")[0] + "</think>\n\n"
        else:
            logger.info("Reasoning hit its token budget. forcing closure.")
            reasoning_text = f"<think>{reasoning_text.split('<think>')[-1]}\n</think>\n\n"

        inputs2 = self.tokenizer(prompt + reasoning_text, return_tensors="pt").to(self.model.device)
        try:  # Phase 2: answer
            with torch.inference_mode():
                out2 = self.model.generate(
                    **inputs2,
                    max_new_tokens=answer_budget,
                    do_sample=True,
                    temperature=0.8,
                    repetition_penalty=1.1,
                    top_p=0.9,
                )
            answer_text = self.tokenizer.decode(out2[0, inputs2["input_ids"].shape[1] :], skip_special_tokens=True)
        finally:
            del inputs2
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        clean_answer = re.sub(r"<think>[\s\S]*?</think>", "", answer_text)
        clean_answer = re.sub(r"</?think>", "", clean_answer)
        clean_answer = re.sub(r"^(?:assistant\s*)+", "", clean_answer.strip(), flags=re.IGNORECASE)

        return reasoning_text, clean_answer.strip()

    def generate(self, query: str, format_type: str = "sketch") -> dict:
        """Executes the RAG chain, dynamically injecting instructions and returning full context."""

        FORMAT_MAPPING = {
            "sketch": {
                "system_instruction": "És um argumentista profissional de comédia e sátira portuguesa.",
                "task_instruction": "escreve um novo sketch de comédia para um vídeo entre 2 e 5 minutos sobre o seguinte tema",
            },
            "newspaper": {
                "system_instruction": "És um cronista satírico a escrever um artigo de opinião para um jornal português.",
                "task_instruction": "escreve um texto de opinião humorístico e satírico com entre 5 a 12 parágrafos sobre o seguinte tema",
            },
            "tv_show": {
                "system_instruction": "És o guionista de um programa de televisão humorístico estilo 'late-night' sobre a atualidade portuguesa.",
                "task_instruction": "escreve o guião de um monólogo televisivo de entre 2 a 5 minutos de duração que relata eventos reais de forma cómica sobre",
            },
        }

        instructions = FORMAT_MAPPING.get(format_type, FORMAT_MAPPING["sketch"])

        if self.dry_run:
            return {"text": f"[DRY RUN] Generated mock {format_type} output.", "sources": []}

        try:
            # explicitly retrieve the documents
            docs_with_scores = self.vector_store.similarity_search_with_score(query, k=3)

            SCORE_THRESHOLD = 0.50
            filtered = [(doc, score) for doc, score in docs_with_scores if score <= SCORE_THRESHOLD]

            # extract IDs and actual text
            retrieved_contexts = []
            retrieved_ids = []
            for doc, score in filtered:
                logger.info(
                    f'\tscore={score:.4f} // id={doc.metadata["sketch_id"]} // preview="{doc.metadata["clean_content"][:30]}"'
                )

                s_id = doc.metadata.get("sketch_id", "UNKNOWN")
                s_text = doc.metadata.get("clean_content", "")

                retrieved_ids.append(s_id)
                retrieved_contexts.append({"sketch_id": s_id, "text": s_text})

            docs = [doc for doc, _ in filtered]

            # format the documents into a string context for the LLM Prompt
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
                    {"role": "system", "content": f"{instructions['system_instruction']}\n..."},
                    {
                        "role": "user",
                        "content": f"{formatted_context}\n\nCom base na inspiração acima, {instructions['task_instruction']}: {query}",
                    },
                ]
            else:
                messages = [
                    {"role": "system", "content": f"{instructions['system_instruction']}\n..."},
                    {
                        "role": "user",
                        "content": f"{instructions['task_instruction']}: {query}",
                    },
                ]
                logger.warning(f"No sufficiently relevant sketches found for query: {query[:60]}...")

            reasoning, clean_response = self._generate_bounded(messages)

            return {"text": clean_response, "sources": retrieved_contexts, "reasoning": reasoning}

        except Exception as e:
            logger.error(f"Generation chain failed for prompt: {query[:30]}... Reason: {e}")
            return {"text": "ERROR: Generation failed.", "sources": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ComedIA RAG Batch Generator")
    parser.add_argument("--dry-run", action="store_true", help="Run without loading Qwen model")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    corpus_dir = os.path.normpath(script_dir / "../../data/04_rag_ready")
    input_path = os.path.normpath(script_dir / "input_prompts.json")
    output_path = os.path.normpath(script_dir / "../../data/06_comedia_outputs/ComedIA_RAG_outputs.json")

    print(input_path)

    # load input dataset
    if not input_path:
        logger.error(f"Input file '{input_path}' not found.")
        exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)

    logger.info(f"Loaded {len(prompts_list)} prompt pairs from {input_path}")

    # initialize RAG system
    rag_system = ComediaRAG(corpus_dir=str(corpus_dir), llm_model_name="Qwen/Qwen3.5-9B", dry_run=args.dry_run)

    # batch processing loop
    output_data = []
    total_items = len(prompts_list)

    for idx, item in enumerate(prompts_list, start=1):
        theme = item.get("theme", "general")
        format_type = item.get("format", "sketch")
        prompt = item.get("prompt", "")

        logger.info(f"Processing ({idx}/{total_items}) // Theme: '{theme}' // Format: '{format_type}'")

        generation_result = rag_system.generate(query=prompt, format_type=format_type)

        output_data.append(
            {
                "theme": theme,
                "format": format_type,
                "prompt": prompt,
                "retrieved_context": generation_result["sources"],
                "output": generation_result["text"],
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Batch generation complete! Saved {len(output_data)} generations to: {output_path}")
