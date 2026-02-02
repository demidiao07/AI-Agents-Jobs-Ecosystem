# Generative AI Assignment 1
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
  chroma_smoketest.py
  chunk_smoketest.py
  embedding_smoketest.py
  ingest_to_chroma.py
  holdout_retrieval_test.py
  retrieval_examples.py
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
The dataset **AI_Agents_Ecosystem_2026.csv** contains structured information about emerging AI agent roles, required skills, tools, and industry applications observed primarily in 2025–2026. The data focuses on modern agentic workflows, multi-agent orchestration frameworks (e.g., LangGraph), and applied use cases across finance, software engineering, and enterprise automation.

### Dataset Scope Filtering
To ensure the dataset is outside the base language model’s knowledge scope, the ingestion pipeline filters records to dates after October 2023. This ensures that the majority of the content reflects ecosystem developments that occurred after common LLM training cutoffs.

### Why the Dataset Is Out of Scope of the LLM
The language models and embedding models used in this project have a knowledge cutoff in late 2023. Many of the roles, tools, and workflows represented in this dataset emerged after that cutoff, meaning the base model cannot reliably answer questions about them without retrieval.

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

```md
### Part 1
# Q1: ChromaDB smoketest
python src/chroma_smoketest.py

# Q3: Chunking + chunk experiments
python src/chunk_smoketest.py

# Q4: Embedding smoketest
python src/embedding_smoketest.py

# Q5: Ingest (chunk + embed + store in Chroma)
python src/ingest_to_chroma.py

# Q6/Q7: Retrieval validation/examples
python src/holdout_retrieval_test.py
python src/retrieval_examples.py

### Part 2
# Q2: Baseline vs Simple RAG
python src/basic_rag_langchain.py

# Q3/Q4: Baseline vs Simple RAG vs HyDE vs Rerank (5 questions)
python src/rag_hyde_rerank.py
```
### Q1: ChromaDB smoketest

I chose ChromaDB as my vector database because it is free, open-source, and supports persistent local storage. I created a persistent client, created a collection, inserted sample documents with metadata, and verified similarity-based retrieval using a test query.

```bash
python src/chroma_smoketest.py
```

### Q3: Chunking smoketest + chunk size experiments

```bash
python src/chunk_smoketest.py
```

I tested chunking configurations:
- **350 / 50**: many small chunks (risk of fragmented context)
- **1000 / 150**: fewer large chunks (risk of retrieving irrelevant context)
- **700 / 100**: balanced retrieval specificity and contextual completeness
Based on these experiments, **700 / 100** was selected for ingestion.

### Q4: Embedding model smoketest (FastEmbed)

I tested the FastEmbed model `BAAI/bge-small-en-v1.5` by embedding semantically related and unrelated texts and computing cosine similarity. Related texts consistently exhibited higher similarity scores than unrelated texts, indicating that the embedding space captures semantic relationships effectively.

```bash
python src/embedding_smoketest.py
```

### Q5: Ingest chunk embeddings into ChromaDB

I chunked each document using LangChain’s `RecursiveCharacterTextSplitter` (700 characters with 100 overlap), embedded all chunks using OpenAI’s `text-embedding-3-small`, and stored vectors with associated metadata in a persistent ChromaDB collection.

```bash
python src/ingest_to_chroma.py
```

### Q6: Retrieval validation (OpenAI-embedded queries)

I validated retrieval quality by embedding representative queries using the same model (`text-embedding-3-small`, 1536 dimensions) and retrieving the nearest neighbors from ChromaDB. For each query, the top-ranked chunks were thematically aligned with the query topic (e.g., reinforcement learning roles, multi-agent orchestration, tool-calling agents), indicating that the vector index behaves as expected.

```bash
python src/holdout_retrieval_test.py
```

### Q7: Additional retrieval examples

I ran five representative natural-language queries and retrieved the top-5 most similar chunks from ChromaDB. Returned results included metadata (title, source, date, link) to support traceability and future citation in a RAG pipeline.

```bash
python src/retrieval_examples.py
```


For Part 2, I used OpenAI’s gpt-4o-mini as the language model for answer generation. The model was accessed via the OpenAI API using the provided openai.txt key on DeepDish, ensuring no API keys were committed to the repository. A smaller model was chosen to stay within usage limits while remaining sufficient for RAG-based response synthesis.

Question 2:The baseline LLM produced a general description of multi-agent orchestration and speculative statements about LangGraph, without any ability to reference the dataset. In contrast, the Simple RAG system retrieved relevant dataset passages and generated a grounded answer citing specific entries (e.g., latency-aware orchestration for parallel multi-agent systems and an orchestration layer for explainable agents). This demonstrates that retrieval improves factual grounding and reduces hallucination by restricting the answer to retrieved evidence.

Question 4:
For each of five representative questions, we evaluated four configurations: (1) a baseline LLM without retrieval, (2) simple RAG using vector similarity search, (3) RAG with HyDE-based query rewriting, and (4) RAG with LLM-based reranking.
The baseline LLM frequently hallucinated or relied on outdated pre-2023 knowledge. Simple RAG substantially improved factual grounding but sometimes retrieved loosely related documents. HyDE improved recall by expanding the query into a hypothetical answer, often surfacing more semantically aligned documents. Reranking produced the most focused responses by selecting passages most relevant to the query intent, particularly for nuanced questions involving safety, evaluation, and tool usage.

Q5: Effect of RAG Variants on Output Quality
This section analyzes how each retrieval strategy affects answer quality for the task of querying a post-2023 AI agent ecosystem dataset, which is out of scope for the base language model.
The goal is to produce accurate, grounded, and traceable answers to questions about tools, agent frameworks, safety, evaluation, and deployment practices.

Original LLM (No RAG)
Observed behavior
* Generated plausible but hallucinated tools, frameworks, and citations (e.g., JADE, ROS, Kafka).
* Relied on generic, pre-2023 knowledge.
* Could not reference dataset-specific titles, dates, or links.
Impact
* Fails to meet the task requirements due to lack of grounding.
* Demonstrates that the base LLM alone cannot answer questions about this dataset reliably.
Conclusion A non-RAG baseline is insufficient because the dataset contains recent (2025–2026) information unavailable to the model at training time.

Simple RAG (Vector Retrieval + Top-K)
Observed improvements
* Answers became grounded in actual dataset entries (e.g., 2026 arXiv papers, HackerNews sources).
* Outputs included source titles, dates, and links, enabling traceability.
* Hallucinations were significantly reduced.
Limitations
* Retrieval sometimes returned partially relevant or noisy documents, especially given the mixed nature of the dataset (jobs, research papers, forum posts).
* Pure similarity-based retrieval occasionally missed the most relevant passages.
Conclusion Simple RAG establishes factual grounding but is sensitive to retrieval noise and query phrasing.

RAG + HyDE (Hypothetical Document Embeddings)
Observed improvements
* HyDE improved recall, especially for abstract or underspecified questions.
* Particularly effective for:
    * Tool-integrated reasoning
    * Agent evaluation and safety themes
    * Multi-agent orchestration concepts
* Surfaced relevant 2026 sources that simple RAG sometimes missed.
Tradeoffs
* Occasionally retrieved conceptually related but less focused passages.
* Depends on the quality of the LLM-generated hypothetical answer.
Conclusion HyDE improves coverage and robustness when querying a rapidly evolving domain with heterogeneous documents.

RAG + Reranking (LLM-Based Passage Selection)
Observed improvements
* Produced the most precise and focused answers across all evaluated questions.
* Effectively filtered out marginally relevant candidates from a large retrieval set.
* Strong performance on:
    * Tool-calling and safety questions
    * Deployment constraints (latency, access control, monitoring)
    * Fine-grained analytical queries
Cost
* Higher latency and compute due to LLM reranking.
* Requires careful prompt design.
Conclusion Reranking yields the highest answer quality by prioritizing relevance over raw similarity, making it the best-performing approach for complex analytical questions.

Overall Comparison
Method	Grounded in Dataset	Recall	Precision	Notes
Original LLM	❌	❌	❌	Hallucinates
Simple RAG	✅	⚠️	⚠️	Baseline grounding
RAG + HyDE	✅	✅	⚠️	Better recall
RAG + Reranking	✅	✅	✅	Best overall
Final Takeaway
Each enhancement addresses a different failure mode of retrieval-augmented generation:
* Simple RAG provides grounding.
* HyDE improves recall for ambiguous or abstract queries.
* Reranking maximizes relevance and answer quality.
For this dataset and task, RAG with reranking consistently produced the strongest results, while HyDE offered meaningful improvements over basic retrieval in exploratory queries.


## Sample Outputs
See `sample_outputs.ipynb` for:
- chunking experiments with different chunk sizes
- embedding dimension sanity checks
- ChromaDB retrieval examples with top-k neighbors
