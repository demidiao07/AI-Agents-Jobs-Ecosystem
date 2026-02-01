# generative-ai-rag

Vector DB (Q1)
* We use ChromaDB as a local persistent vector database.
* Run:

```bash
pip install -r requirements.txt
python src/vector_db/q1_chroma_smoketest.py
```

**Dataset Selection:**

For this project, we use the Exorde Social Media December 2024 (Week 1) dataset, which contains social media posts collected during the first week of December 2024. The dataset provides short-form textual content reflecting real-time discussions of current events, news, and public opinion.

**LLM Selection:**

The LLM selected for the Retrieval-Augmented Generation (RAG) pipeline is GPT-4o mini. This model provides sufficient capacity for in-context learning, reasoning, and retrieval-augmented question answering, while remaining lightweight and cost-efficient for experimentation.

**Out-of-Scope Justification:**

GPT-4o mini has a knowledge cutoff of October 2023. Since the Exorde dataset contains social media posts created in December 2024, the information is outside the model’s parametric knowledge. As a result, the model cannot rely on memorized facts and must retrieve relevant information from an external vector database, making the dataset well-suited for RAG-based workflows.

**Reusability for Future Assignments:**

This dataset is compatible with future course assignments because it can support multiple downstream tasks, including:
* Retrieval-augmented question answering
* Topic clustering and trend detection
* Summarization of evolving events
* Few-shot prompting with labeled examples
* LoRA fine-tuning using human-labeled data such as sentiment or topic annotations
  
The short, independent nature of social media posts also simplifies manual or semi-automated labeling in later fine-tuning tasks.


Chunking was performed using LangChain’s RecursiveCharacterTextSplitter. Since social media posts are generally short, most posts were treated as single chunks. Longer posts were split using a chunk size of 512 characters with an overlap of 50 characters. Chunking behavior was validated through exploratory analysis in a Jupyter notebook.


Embedding models were evaluated in notebooks/embedding_experiments.ipynb using nearest-neighbor sanity checks. We selected all-MiniLM-L6-v2 for the remainder of the project.
