# Generative AI Assignment 1 – Part 1
**Name:** Wenshu (Demi) Diao

## Overview
This project implements a vector database pipeline for AI agent–related content using:
- document chunking
- text embeddings
- ChromaDB for vector storage
- similarity-based retrieval

## Project Structure

```markdown
src/
ingest_to_chroma.py
chunk_smoketest.py
embedding_smoketest.py
retrieval_examples.py
...
requirements.txt
sample_outputs.ipynb
README.md
.gitignore
```

## Setup Instructions
### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API key
Set the API key as an environment variable (I do not upload my API key):

```
export OPENAI_API_KEY="your_api_key_here"
```

## Dataset
The dataset **AI_Agents_Ecosystem_2026.csv** contains structured information about emerging AI agent roles, required skills, tools, and industry applications observed in 2025–2026. The data focuses on modern agentic workflows, multi-agent orchestration frameworks, and applied use cases across finance, software engineering, and enterprise automation.

### Why the Dataset Is Out of Scope of the LLM
The language models and embedding models used in this project have a knowledge cutoff in late 2023. Many of the roles, tools, and ecosystem developments represented in this dataset emerged after that cutoff, making the information unavailable to the base model without retrieval.

### Why RAG Is Appropriate 
Because the dataset contains recent, domain-specific information that evolves rapidly, retrieval-augmented generation is necessary to ground model responses in up-to-date facts and reduce hallucinations when answering questions about modern AI agent ecosystems.

**Dataset source (Kaggle):**  
👉 [AI Agents Jobs Ecosystem 2026 – Real World](https://www.kaggle.com/datasets/nudratabbas/ai-agents-jobs-ecosystem-2026-real-world)

**Note:** The dataset is **not included** in this repository in accordance with assignment instructions.

After downloading, place the CSV file at:
```text
data/ai_agents_jobs/AI_Agents_Ecosystem_2026.csv
```

## How to Run 

### Q1: ChromaDB smoketest

I chose ChromaDB as my vector database because it is free, open-source, and supports persistent local storage. I created a persistent client, created a collection for my dataset, inserted sample documents with metadata, and verified retrieval by running a query. This collection will store embedded chunks of my dataset for the RAG pipeline.

```bash
python src/chroma_smoketest.py
```

### Q3: Chunking smoketest + chunk size experiments
```bash
python src/chunk_smoketest.py
```

### Q4: Embedding model smoketest (FastEmbed)
```bash
python src/embedding_smoketest.py
```

### Q5: Ingest chunk embeddings into ChromaDB
```bash
python src/ingest_to_chroma.py
```

### Q6: Retrieval validation (OpenAI-embedded queries)

I validated retrieval quality by querying the ChromaDB collection using embeddings from the same model used for ingestion (text-embedding-3-small, 1536 dimensions). For each test query, the top retrieved chunks were thematically aligned with the query topic (e.g., reinforcement learning papers for RL queries, orchestration results for multi-agent orchestration, and tool-safety/function-calling results for tool-calling queries). This indicates that the embedding model + vector database return nearest neighbors that are meaningful in the original text space.

```bash
python src/holdout_retrieval_test.py
```

### Q7: Additional retrieval examples

I queried the persisted ChromaDB collection using multiple natural-language queries embedded with the same model used during ingestion (text-embedding-3-small). For each query, the top-k retrieved chunks were semantically aligned with the query topic, demonstrating consistent and meaningful nearest-neighbor retrieval behavior in the vector database.

```bash
python src/retrieval_examples.py
```

## Sample Outputs
See `sample_outputs.ipynb` for:
- chunking experiments with different chunk sizes
- embedding dimension sanity checks
- ChromaDB retrieval examples with top-k neighbors
