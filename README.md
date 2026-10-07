# Luso-Laugh / ComedIA

This repository holds the code for my Master's thesis at IST on generating Portuguese (pt-PT) satire with LLMs. It does two things:

1. **Luso-Laugh** builds a corpus of Portuguese comedy sketches from YouTube: download the audio, transcribe it, find where the audience laughs, and annotate why those lines are funny.
2. **ComedIA** uses that corpus to compare ways of getting Qwen3.5-9B to write better Portuguese comedy: the plain model, retrieval (RAG), LoRA fine-tuning, and a mix of both. An LLM-as-a-judge panel then scores everything.

The current corpus has **556 sketches, about 39 hours of audio**. They come from Gato Fedorento, Herman José (2010–2013), *Isto é Gozar com Quem Trabalha*, *Cá Por Casa*, *Ruído* and a few others.

Everything runs as plain Python scripts, one per step. The letter at the start of each file name (A, B, C, …) is roughly the order you run them in.

---

## Repository layout

```
.
├── config/
│   ├── source_channels.csv      # YouTube channels/playlists to scrape (url, fallback_language)
│   └── source_videos.csv        # individual videos, optionally with start/end cut points
├── data/
│   └── 01_catalogs/
│       └── luso_laugh_catalog.csv   # one row per sketch, produced by step A
├── scripts/                     # small shell helpers (see "Shell scripts" below)
├── src/
│   ├── common/                  # prompts and generation settings shared by all generators
│   ├── 01_data_building/        # A: catalog, B: audio download
│   ├── 02_data_process/         # C: transcription + annotation, D: RAG/LoRA datasets
│   ├── 03_llm_evaluation/       # base model selection (which LLM to build on)
│   ├── 04_training/             # the ComedIA architectures: baseline, RAG, LoRA, hybrid
│   └── 05_evaluation/           # LLM-as-a-judge evaluation, gold standard, audio rendering
└── requirements.txt
```

Only `data/01_catalogs` ships with the code. Every other `data/` folder is created by the scripts as you go.

---

## Setup

**Python and packages.** I used Python 3.10. `requirements.txt` lists all the required packages.
```

You also need `ffmpeg` on the system for `yt-dlp`.

**GPU.** Everything that touches Qwen3.5-9B (steps C and D, and every generator and trainer in `04_training`) needs a CUDA GPU. The model runs in bf16, so plan for roughly 24 GB of VRAM to generate and more to train. I trained on an A40. Most scripts also run on a V100, though more slowly.

**Environment variables.** Create a `.env` file in the repository root:

```
HF_TOKEN=...              # Hugging Face, needed for Qwen and the pyannote diarization models

# base model selection (03_llm_evaluation), any OpenAI-compatible endpoint
EVAL_BASE_URL=...
EVAL_API_KEY=...
EVAL_MODEL=...

# final evaluation judges (05_evaluation)
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
XAI_API_KEY=...
```

For diarization to work, accept the pyannote model terms on Hugging Face with the account behind `HF_TOKEN`.

> `.env` contains real keys. `scripts/zip_project.sh` currently adds it to the zip, so take it out before sharing the project with anyone.

---

## Running the pipeline

Run commands from the repository root unless a step says otherwise. Most scripts work out their paths from their own location, so the working directory rarely matters. The exceptions are noted where they apply.

### Step 1: Build the catalog (`A_build_catalog.py`)

```bash
python src/01_data_building/A_build_catalog.py
```

Reads `config/source_channels.csv` and `config/source_videos.csv` and uses `yt-dlp` to list every video, without downloading anything yet. The result is `data/01_catalogs/luso_laugh_catalog.csv`:

| column | meaning |
|---|---|
| `sketch_id` | YouTube video ID, used as the sketch's ID everywhere downstream |
| `title`, `url`, `channel_name` | taken from YouTube |
| `start_time`, `end_time` | seconds to keep. Defaults to the whole video unless `source_videos.csv` gives cut points |
| `declared_language` | YouTube's language tag, or the fallback from the config |
| `status` | `pending` → `downloaded` → `processed`, updated by the next two steps |

To add material, add a row to one of the config files and run this again. Duplicate IDs are dropped.

### Step 2: Download the audio (`B_download_audio.py`)

```bash
python src/01_data_building/B_download_audio.py
```

Downloads every `pending` sketch as an MP3 into `data/02_audio_corpus/`, cut to `start_time`–`end_time`. Each success marks the row as `downloaded`. Failed downloads are retried three times with backoff. You can stop and restart it at any point.

### Step 3: Transcribe and annotate (`C_data_pipeline.py`)

```bash
python src/02_data_process/C_data_pipeline.py
```

This is the heavy step. For every `downloaded` sketch it does four things:

1. **Detects laughter.** An AudioSet classifier (`MIT/ast-finetuned-audioset`) slides a 3-second window over the audio and flags windows where it detects laughter, giggling or chuckling. Overlapping windows are merged into laughter events.
2. **Transcribes and diarizes.** WhisperX `large-v3` transcribes the audio, aligns it word by word, and labels the speakers as `SPEAKER_00`, `SPEAKER_01`, and so on.
3. **Marks punchlines.** A line counts as a punchline if a laughter event starts during it or within 2 seconds after it ends.
4. **Annotates the humour.** For every punchline, Qwen3.5-9B gets the dialogue since the previous punchline and writes a short academic explanation of why the line is funny.

The output is `data/03_final_dataset/<sketch_id>_annotated.json`: a list of segments with `speaker`, `text`, `start`, `end`, `is_punchline` and `semantic_metadata.humor_analysis`. The catalog row then moves to `processed`.

There is a Demucs vocal-separation step in the code (`separate_sources`), but it is commented out in `process_sketch_remote`, so the original mix is what gets transcribed.

### Step 4: Build the RAG corpus and the LoRA dataset (`D_data_transform.py`)

This script imports from `src/common`, so it needs `src` on the path:

```bash
PYTHONPATH=src python src/02_data_process/D_data_transform.py
```

Options:
- `--limit N` processes only N sketches, for testing.
- `--dry-run` skips the LLM.
- `--macro-variants N` writes N training examples per sketch; the default is 1.

For each annotated sketch:

1. **Script enrichment.** Consecutive lines from the same speaker are merged into one turn. Qwen then rewrites the transcript as a script with real character names (`[Anfitrião]:` instead of `[SPEAKER_00]:`) and the occasional stage direction.
   - The rewrite must keep the original words. If it changes the length by more than 40%, loses turns, or still has `SPEAKER_NN` tags, the raw transcript is kept instead.
   - In the current dataset, 501 of the 556 sketches (about 90%) were enriched.
2. **RAG document.** `data/04_rag_ready/<sketch_id>_rag.json` holds three fields:
   - `content`: the script.
   - `comedic_metadata`: all the punchline analyses.
   - `sketch_id`.
3. **LoRA example.** Each example is one line in `data/04_lora_ready/lora_instruction_dataset.jsonl`:
   - `instruction`: the writing task.
   - `input`: a synthetic premise that Qwen wrote from the sketch.
   - `output`: a `<think>` planning block followed by the script. Qwen builds the plan backwards from the real script and its punchlines (cast, rejected approach, comic arc, escalation).

The script skips sketches that already have a `_rag.json`, so you can run it again.

Optional check:

```bash
python src/02_data_process/aux_check_tokens.py --limit 4096
```

This tells you how many LoRA examples are too long for the training context, and would therefore be truncated.

### Step 5: Choose the base model (`03_llm_evaluation/`)

You only need this to reproduce how Qwen3.5-9B was chosen. It is a smaller, older evaluation:

```bash
python src/03_llm_evaluation/D_generation.py        # each candidate writes 4 sketches
python src/03_llm_evaluation/E_evaluation.py        # 9 personas score them through the EVAL_* endpoint
python src/03_llm_evaluation/aux_final_evaluation.py  # ranks the models in final_evaluation_results.csv
```

Some notes on these three scripts:
- **Uncomment the candidates first.** All of them in `CANDIDATE_MODELS` (`D_generation.py`) are commented out, so uncomment the ones you want to test.
- **Run it from the repository root.** `D_generation.py` writes to `data/04_eval_generations/` relative to where you launch it.
- **The judge is any OpenAI-compatible endpoint.** I used Groq.
- **Outputs:** generations go to `data/04_eval_generations/` and scores to `data/05_eval_results/`.
- **This stage used 9 personas.** The final evaluation (step 7) dropped the Public Relations Specialist and uses 8.

### Step 6: Generate with each architecture (`04_training/`)

Every generator reads the 15 prompts in `src/04_training/input_prompts.json`. There are 11 sketches, 2 newspaper columns and 2 TV monologues, on themes such as bureaucracy, football, housing and the Algarve. Each generator writes `data/06_comedia_outputs/ComedIA_<name>_outputs.json`.

All of them share the prompts and decoding settings in `src/common/`:

- **System prompt and instructions.** `prompts.py` has the system prompt and one instruction per format: sketch, newspaper chronicle, monologue.
- **Generation runs in two passes.**
  - First a reasoning pass. It stops when the model closes `</think>`, or at 5,000 tokens.
  - Then an answer pass of up to 2,250 tokens.
- **Repetition control.** A windowed repetition penalty (`RepetitionControlProcessor`) stops the model looping.

| Architecture | Command | Output |
|---|---|---|
| Baseline (plain Qwen) | `python src/04_training/F_comedia_BASE.py` | `ComedIA_Baseline` |
| RAG, dense (bge-m3 + Chroma, k=3, cosine distance ≤ 0.46) | `python src/04_training/F_comedia_RAG.py` | `ComedIA_RAG` |
| … content only, without the punchline metadata | `... F_comedia_RAG.py --no-metadata` | `ComedIA_RAG_NM` |
| RAG, BM25 (k=3) | `python src/04_training/F_comedia_RAG_bm25.py` | `ComedIA_RAG_BM25` |
| … content only | `... F_comedia_RAG_bm25.py --no-metadata` | `ComedIA_RAG_BM25_NM` |
| LoRA: train | `python src/04_training/F_comedia_PEFT.py --mode train --rank 16` | adapter in `src/04_training/adapters/` |
| LoRA: generate | `python src/04_training/F_comedia_PEFT.py --mode generate --rank 16` | `ComedIA_PEFT_r16` |
| Hybrid (LoRA + dense RAG) | `python src/04_training/F_comedia_MIXED.py --rank 16` | `ComedIA_Hybrid_r16` |

**LoRA details.**
- **Ranks:** `--rank` takes 4, 8, 16, 32 or 64, with alpha = 2 × rank.
- **Training settings:** batch size 1 with 8 accumulation steps, learning rate 1e-4, cosine schedule, up to 10 epochs, bf16.
- **Data split:** 10% of the LoRA dataset is held out for evaluation.
- **Checkpoints:** saved every 20 steps under `adapters/checkpoints_r<rank>/`. The best one (lowest eval loss) is kept as `adapters/comedia_lora_r<rank>`.
- **Loss masking:** the `<think>` block stays in the input but is excluded from the loss (`MaskThinkCollator`).
- **Base-vs-adapter check:** at the end of training the script compares the adapter's eval loss with the base model's and saves the result to `adapters/eval_r<rank>.json`.
- **Training curves:** draw them with:

```bash
python src/04_training/aux_training_report.py --target_dir src/04_training/adapters/checkpoints_r16 --output r16_report.png
```

- **Generating from a specific checkpoint:** use `--adapter-path adapters/checkpoints_r16/checkpoint-120`.

**Retrieval sweep.** `F_comedia_RAG_sweep.py` is how I picked k=3 and the 0.46 threshold for dense RAG:
- **`--preview`** prints the retrieval distances for every prompt without generating anything.
- **`--sweep k|threshold|both`** generates one output file per setting.
- **Outputs:** the files go into `06_comedia_outputs` as `ComedIA_outputs_k*.json` and `..._t*.json`. **Move them somewhere else before step 7**, or they get evaluated too.

All generators accept `--dry-run`, which runs the loop without loading the model.

### Step 7: Evaluate (`05_evaluation/`)

**7a. The human gold standard.** `G_eval_set.py` takes 11 hand-picked sketches from the LoRA dataset (`SELECTED_SKETCHES`) and writes two files:
- `ComedIA_Human_outputs.json`: their premise, their script, and an empty `edited_output` for you to fill in.
- `eval_dataset_prompts.json`: the premises alone.

```bash
python src/05_evaluation/G_eval_set.py
```

Then fill in `edited_output` by hand. The transcripts are noisy, so fix the speaker names and obvious transcription errors without changing the jokes. Copy the finished file into `data/06_comedia_outputs/` so it gets evaluated alongside the models. These 11 sketches have different premises from the 15 prompts, so treat them as a reference level, not a paired comparison.

**7b. LLM-as-a-judge.**

```bash
python src/05_evaluation/F_evaluation.py --judge luna      # or sol, sonnet, grok
```

**What it evaluates.** Every `.json` in `data/06_comedia_outputs/`. Each text is scored by:
- **8 personas.** Author/Novelist, Content Writer, Technical Writer, Copywriter, Editor, Comedian, Journalist and Portuguese (Portugal) Specialist. They are adapted from Kim and Oh (2025).
- **3 repeats per persona.**
- **7 metrics, each from 0 to 4:** novelty, clarity, relevance, intelligence, empathy, cultural resonance and overall score. Brazilian Portuguese is explicitly penalised under cultural resonance.

**The judges** are set in `JUDGES` at the top of the file:

| `--judge` | model | settings |
|---|---|---|
| `luna` | GPT-6 Luna | medium reasoning effort |
| `sol` | GPT-6 Sol | medium reasoning effort |
| `grok` | Grok 4.3 | medium reasoning effort, through xAI's OpenAI-compatible API |
| `sonnet` | Claude Sonnet 5 | adaptive thinking, low effort |

**Running it.** Each judge writes to its own folder, `data/07_eval_results/<judge>/`, so all four can run in parallel. A full run is about 3,900 API calls per judge. Try `--limit 1` first, which evaluates only the first item of every file, to check that the keys and costs are what you expect.

**What each judge folder contains:**

| file | contents |
|---|---|
| `<arch>_raw_judgements.jsonl` | every single API response, one per line |
| `<arch>_individual_personas.csv` | the 3 repeats averaged, one row per item and persona |
| `<arch>_per_item.csv` | the 8 personas averaged, one row per item |
| `Master_Aggregated_Results.csv` | one row per architecture: the mean of each metric over items, its spread across items (`_sd`), persona disagreement, repeat consistency and mean word count |
| `Diagnostics.md` | the judge's own summary of each architecture's strengths and weaknesses, in Portuguese |

**7c. Audio (optional).**

```bash
python src/05_evaluation/G_audio_output.py --limit 1
```

Reads the outputs in `06_comedia_outputs` and reads them aloud with Chatterbox's pt-PT model. Each character gets its own voice, built from an Edge TTS sample. Qwen first adds light paralinguistic tags such as `[laugh]` and `[sigh]`; `--no-tags` turns that off. The `.wav` and `.txt` files go to `data/07_audio_eval/`.

---

## Data folders at a glance

| folder | created by | contents |
|---|---|---|
| `01_catalogs/` | A | the sketch catalog CSV |
| `02_audio_corpus/` | B | one MP3 per sketch |
| `03_final_dataset/` | C | annotated transcripts (`*_annotated.json`) |
| `04_rag_ready/` | D | one RAG document per sketch (`*_rag.json`) |
| `04_lora_ready/` | D | `lora_instruction_dataset.jsonl` |
| `04_eval_generations/`, `05_eval_results/` | step 5 | base model selection (old, separate stage) |
| `06_comedia_outputs/` | step 6 | one JSON per architecture, plus the human gold standard |
| `07_eval_results/<judge>/` | step 7b | evaluation results per judge |
| `07_audio_eval/` | step 7c | rendered audio |

---

## Shell scripts

- **`scripts/make_data.sh`** wipes `data/` and recreates the early folders. It deletes everything in `data/`, including the catalog, so be careful with it. It still creates the old folder names; the newer ones are created by the Python scripts themselves.
- **`scripts/zip_project.sh`** and **`scripts/zip_data.sh`** zip the code (with the catalog) or the whole `data/` folder, for moving between my machine and the cluster.
- **`scripts/clean_cluster.sh`** deletes the whole project from the current directory. It's meant for cleaning the cluster after a run. Don't run it on your own copy.

---

## Things worth knowing

- **Outputs are overwritten.** Every generator overwrites its output file, and the evaluation overwrites its judge's folder. Copy anything you want to keep first.
- **Seeds are fixed.** Generation uses seed 42, but sampling is on, so runs on different hardware will not match token for token.