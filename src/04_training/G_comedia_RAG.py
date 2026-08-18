import argparse
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
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


class ComediaRAG:
    """
    Implements Architecture A (RAG) using LangChain, ChromaDB, and local Qwen-3.5-9B.
    """

    def __init__(
        self,
        corpus_dir: str = "../../data/04_rag_ready",
        llm_model_name: str = "Qwen/Qwen3.5-9B",
        embedder_model_name: str = "rufimelo/bert-large-portuguese-cased-sts",
        dry_run: bool = False,
    ):
        self.corpus_dir = Path(corpus_dir)
        self.dry_run = dry_run

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = torch.bfloat16 if self.device == "cuda" else torch.float32

        self.hf_token = os.getenv("HF_TOKEN")

        self.vector_store = None
        self.llm_chain = None

        self._build_vector_store(embedder_model_name)
        self._init_llm_chain(llm_model_name)

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
        embeddings = HuggingFaceEmbeddings(
            model_name=embedder_model_name,
            model_kwargs={"device": self.device},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("Ingesting documents into ephemeral ChromaDB...")
        self.vector_store = Chroma.from_documents(documents=lc_documents, embedding=embeddings)

    def _init_llm_chain(self, llm_model_name: str):
        """Initializes Qwen-3.5-9B and constructs a decoupled LCEL chain."""
        if self.dry_run:
            logger.info("DRY RUN: Skipping Qwen LLM initialization.")
            return

        logger.info(f"Loading Generative LLM: {llm_model_name}...")
        try:
            hf_pipeline = transformers.pipeline(
                "text-generation",
                model=llm_model_name,
                dtype=self.compute_type,
                device_map="auto",
                token=self.hf_token,
                max_new_tokens=4098,
                temperature=0.8,
                do_sample=True,
                return_full_text=False,
            )

            langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)

            template = """<|im_start|>system
                        {system_instruction} 
                        Abaixo estão exemplos de humor português para servirem de inspiração estilística. 
                        Usa o mesmo tom, ritmo, ironia e vocabulário para escrever o novo texto.

                        IMPORTANTE: Responde APENAS com o texto final pedido. Não incluas o teu processo de raciocínio, notas, introduções ou tags internas.<|im_end|>
                        <|im_start|>user
                        {context}

                        Com base na inspiração acima, {task_instruction}: {query}<|im_end|>
                        <|im_start|>assistant
                        """
            prompt = PromptTemplate.from_template(template)
            self.llm_chain = prompt | langchain_llm | StrOutputParser()

        except Exception as e:
            logger.error(f"Failed to load Generative LLM: {e}")
            self.llm_chain = None

    def generate_satire(self, query: str, format_type: str = "sketch") -> dict:
        """Executes the RAG chain, dynamically injecting instructions and returning full context."""

        FORMAT_MAPPING = {
            "sketch": {
                "system_instruction": "És um argumentista profissional de comédia e sátira portuguesa.",
                "task_instruction": "escreve um novo sketch de comédia para um vídeo entre 2 e 5 minutos sobre o seguinte tema",
            },
            "newspaper": {
                "system_instruction": "És um cronista satírico a escrever um artigo de opinião para um jornal português.",
                "task_instruction": "escreve um texto de opinião humorístico e satírico com entre 5 a 15 parágrafos sobre o seguinte tema",
            },
            "tv_show": {
                "system_instruction": "És o guionista de um programa de televisão humorístico estilo 'late-night' sobre a atualidade portuguesa.",
                "task_instruction": "escreve o guião de um monólogo televisivo de entre 2 a 5 minutos de duração que relata eventos reais de forma cómica sobre",
            },
        }

        instructions = FORMAT_MAPPING.get(format_type, FORMAT_MAPPING["sketch"])

        if self.dry_run or not self.llm_chain:
            return {"text": f"[DRY RUN] Generated mock {format_type} output.", "sources": []}

        try:
            # explicitly retrieve the documents
            retriever = self.vector_store.as_retriever(search_kwargs={"k": 3})
            docs = retriever.invoke(query)

            # extract IDs and actual text
            retrieved_contexts = []
            retrieved_ids = []
            for doc in docs:
                s_id = doc.metadata.get("sketch_id", "UNKNOWN")
                s_text = doc.metadata.get("clean_content", "")

                retrieved_ids.append(s_id)
                retrieved_contexts.append({"sketch_id": s_id, "text": s_text})

            logger.info(f"   -> Retrieved source sketches: {retrieved_ids}")

            # format the documents into a string context for the LLM Prompt
            formatted_context = ""
            for i, doc in enumerate(docs, 1):
                formatted_context += (
                    f"--- EXEMPLO {i} ({doc.metadata['sketch_id']}) ---\n{doc.metadata['clean_content']}\n"
                )

            response = self.llm_chain.invoke(
                {
                    "context": formatted_context,
                    "query": query,
                    "system_instruction": instructions["system_instruction"],
                    "task_instruction": instructions["task_instruction"],
                }
            )

            # clean thinking tags and artifacts
            clean_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            clean_response = clean_response.replace("assistant\n", "").strip()

            return {"text": clean_response, "sources": retrieved_contexts}

        except Exception as e:
            logger.error(f"Generation chain failed for prompt: {query[:30]}... Reason: {e}")
            return {"text": "ERROR: Generation failed.", "sources": []}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LangChain Luso-Laugh Batch RAG Generator")
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

        generation_result = rag_system.generate_satire(query=prompt, format_type=format_type)

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
