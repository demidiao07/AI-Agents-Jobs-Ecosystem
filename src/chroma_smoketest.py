import chromadb

# Create a local persistent Chroma DB (stored on disk)
client = chromadb.PersistentClient(path="./chroma_db")

# Create or load a collection
collection = client.get_or_create_collection(name="ai_agents_jobs_2026")

# Add a couple of sample documents (no embeddings yet; Chroma can store text + metadata)
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

# Query (text-based query works if you later add embeddings; for now just test retrieval structure)
results = collection.query(query_texts=["reinforcement learning finance"], n_results=2)

print(results)