# Generative AI Assignment 1
**Name:** Wenshu (Demi) Diao

## Overview
This project implements a complete **vector database and retrieval pipeline** for modern AI agent–related content. The system supports document chunking, text embedding, vector storage, and similarity-based retrieval, and is later extended into a Retrieval-Augmented Generation (RAG) application.

The project is organized into two parts:

- **Part 1**: Vector database construction, chunking, embedding, ingestion, and retrieval validation
- **Part 2**: RAG development and evaluation, including advanced retrieval techniques (HyDE and reranking)

## Key Components
- Document chunking with LangChain
- Text embeddings using OpenAI and FastEmbed models
- ChromaDB for persistent vector storage
- Similarity-based retrieval and validation
- RAG pipelines built with LangChain
- Evaluation of retrieval and generation quality

## Project Structure

```text
src/
  chroma_smoketest.py
  chunk_smoketest.py
  embedding_smoketest.py
  ingest_to_chroma.py
  holdout_retrieval_test.py
  retrieval_examples.py
  basic_rag_langchain.py
  rag_hyde_rerank.py
notebook/
  sample_outputs.ipynb
requirements.txt
README.md
.gitignore
```

## Setup Instructions
### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set OpenAI API key
Set the API key as an environment variable. API keys are **not committed** to the repository.

```
export OPENAI_API_KEY="your_api_key_here"
```

## Dataset
**AI_Agents_Ecosystem_2026.csv** is a structured dataset describing emerging AI agent roles, required skills, tooling, and industry applications observed primarily in **2025–2026**. The dataset emphasizes modern agentic workflows, including multi-agent orchestration frameworks (e.g., LangGraph) and real-world deployments across finance, software engineering, and enterprise automation.

**Dataset source (Kaggle):**  
👉 [AI Agents Jobs Ecosystem 2026 – Real World](https://www.kaggle.com/datasets/nudratabbas/ai-agents-jobs-ecosystem-2026-real-world)

**Note:** The dataset is **not included** in this repository in accordance with assignment instructions.

After downloading, place the CSV file at:
```text
data/ai_agents_jobs/AI_Agents_Ecosystem_2026.csv
```

## Dataset Scope and Motivation for RAG
### Out-of-Scope Justification
To ensure the dataset lies outside the base language model’s knowledge scope, records are filtered to dates **after October 2023**, which is beyond the training cutoff of the LLMs and embedding models used in this project.

Many of the tools, roles, and workflows represented—particularly in agent orchestration, evaluation, and deployment—emerged after this cutoff. As a result, a standalone LLM cannot reliably answer questions about this data without external retrieval.

### Why Retrieval-Augmented Generation Is Necessary
Because the dataset contains recent, rapidly evolving, and domain-specific information, RAG is essential to:
- Ground model responses in up-to-date evidence
- Reduce hallucinations
- Enable traceable, dataset-backed answers

## How to Run 
### Part 1: Vector Database and Retrieval
```bash
# Q1: ChromaDB smoketest
python src/chroma_smoketest.py

# Q3: Chunking experiments
python src/chunk_smoketest.py

# Q4: Embedding smoketest
python src/embedding_smoketest.py

# Q5: Ingest documents into ChromaDB
python src/ingest_to_chroma.py

# Q6/Q7: Retrieval validation and examples
python src/holdout_retrieval_test.py
python src/retrieval_examples.py
```

### Part 2: RAG Evaluation
```bash
# Q2: Baseline LLM vs Simple RAG
python src/basic_rag_langchain.py

# Q3/Q4: Baseline vs Simple RAG vs HyDE vs Reranking
python src/rag_hyde_rerank.py
```

## Part 1: Implementation Details
### ChromaDB Smoketest
ChromaDB was selected because it is free, open-source, and supports persistent local storage. A persistent client and collection were created, sample documents with metadata were inserted, and similarity-based retrieval was verified using test queries.

### Chunking Strategy
Multiple chunking configurations were evaluated:
- **350 / 50**: High fragmentation, reduced context coherence
- **1000 / 150**: Fewer chunks, increased retrieval noise
- **700 / 100**: Balanced specificity and contextual completeness
Based on retrieval behavior, **700 / 100** was selected for ingestion.

### Embedding Evaluation
The FastEmbed model `BAAI/bge-small-en-v1.5` was tested using semantically related and unrelated text pairs. Cosine similarity scores consistently reflected semantic relationships, validating the embedding quality.

For ingestion, OpenAI’s `text-embedding-3-small` (1536 dimensions) was used for consistency between document and query embeddings.

### Retrieval Validation
Representative queries were embedded and used to retrieve nearest neighbors from ChromaDB. Retrieved chunks aligned well with query intent (e.g., multi-agent orchestration, reinforcement learning roles), indicating correct vector index behavior.


## Part 2: RAG Design and Evaluation
OpenAI’s **gpt-4o-mini** was used as the language model to balance capability and API usage limits. All keys were accessed via environment configuration and were not committed to the repository.

### Evaluation Setup
Five representative questions were evaluated across four configurations:
1. Baseline LLM (no retrieval)
2. Simple RAG (vector similarity search)
3. RAG with HyDE query rewriting
4. RAG with LLM-based reranking

### Results Summary
#### Baseline LLM
- Generated plausible but hallucinated tools and frameworks
- Relied on outdated pre-2023 knowledge
- Could not reference dataset-specific entries
#### Simple RAG
- Significantly improved factual grounding
- Included dataset titles, dates, and links
- Occasionally retrieved partially relevant passages
#### RAG + HyDE
- Improved recall for abstract or underspecified queries
- Effective for agent evaluation, safety, and orchestration topics
- Occasionally introduced less focused context
#### RAG + Reranking
- Produced the most precise and relevant answers
- Filtered noisy retrieval candidates effectively
- Higher computational cost due to reranking

In some cases (e.g., Q2 with HyDE), the system correctly abstains when retrieved context is insufficient, demonstrating controlled non-hallucination behavior.

#### Final Takeaway:
Simple RAG establishes grounding, HyDE improves recall, and reranking maximizes relevance. For this dataset and task, RAG with reranking consistently produced the highest-quality results, while HyDE offered meaningful gains over basic retrieval in exploratory queries.

## Sample Outputs
See `sample_outputs.ipynb` for:
- chunking experiments with different chunk sizes
- embedding dimension sanity checks
- ChromaDB retrieval examples with top-k neighbors
