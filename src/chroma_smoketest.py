# Part 1, Question 1: vector DB setup

# Smoke test to verify ChromaDB setup, persistence, and semantic retrieval
# Full embedding + chunking pipeline implemented in ingestion.py

import chromadb
from chromadb.config import Settings

client = chromadb.Client(
    Settings(persist_directory="./chroma_db")
)

collection = client.get_or_create_collection(
    name="ai_agents_jobs_2026"
)

collection.add(
    ids=["doc1", "doc2"],
    documents=[
        "Autonomous AI Agent Engineer in finance uses reinforcement learning and Python.",
        "Multi-agent orchestration roles often use LangGraph and tool calling."
    ],
    metadatas=[
        {"source": "smoketest", "row_id": 1},
        {"source": "smoketest", "row_id": 2}
    ]
)

results = collection.query(
    query_texts=["reinforcement learning finance"],
    n_results=2
)

print(results)
