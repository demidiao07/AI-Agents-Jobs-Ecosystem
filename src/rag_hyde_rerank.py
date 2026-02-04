# Part 2, Question 3: RAG + two advanced elements (HyDE + reranking)

import os
from pathlib import Path
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -----------------------
# Config
# -----------------------
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = BASE_DIR.parent
DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "ai_agents_jobs_2026"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

TOP_K = 5
CANDIDATES_K = 40
RERANK_SNIPPET_CHARS = 350


# -----------------------
# OpenAI key
# -----------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    # optional convenience for course environment; ensure openai.txt is gitignored
    key_path = PROJECT_ROOT / "openai.txt"
    if key_path.exists():
        api_key = key_path.read_text().strip()

if not api_key:
    raise RuntimeError("Missing OPENAI_API_KEY. Please export it in your environment.")

os.environ["OPENAI_API_KEY"] = api_key


# -----------------------
# Vector store
# -----------------------
if not DB_DIR.exists():
    raise RuntimeError(f"Missing {DB_DIR}. Run Part 1 ingestion to build the Chroma DB first.")

embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)

# Note: _collection is internal; useful for debugging in coursework
try:
    count = vectorstore._collection.count()
except Exception:
    count = "unknown"

print("Persist dir:", DB_DIR)
print("Collection:", COLLECTION_NAME)
print("Count:", count)

def retrieve_simple(query: str, k: int = TOP_K):
    """Simple RAG baseline: plain similarity search."""
    return vectorstore.similarity_search(query, k=k)


def retrieve_candidates(query: str, k: int = CANDIDATES_K):
    return vectorstore.max_marginal_relevance_search(
        query,
        k=k,
        fetch_k=max(120, k * 3),
        lambda_mult=0.6
    )


def format_docs(docs: List) -> str:
    """Full context formatter for final answering."""
    parts = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        header = (
            f"[{i}] title={md.get('title','')} | source={md.get('source','')} | "
            f"date={md.get('date','')} | link={md.get('link','')}"
        )
        parts.append(header + "\n" + (d.page_content or ""))
    return "\n\n".join(parts)


def format_docs_for_rerank(docs: List, snippet_chars: int = RERANK_SNIPPET_CHARS) -> str:
    """Short snippets for reranking to keep prompts small."""
    parts = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        text = (d.page_content or "").replace("\n", " ").strip()
        text = text[:snippet_chars]
        header = (
            f"[{i}] title={md.get('title','')} | source={md.get('source','')} | "
            f"date={md.get('date','')} | link={md.get('link','')}"
        )
        parts.append(header + "\n" + text)
    return "\n\n".join(parts)


# -----------------------
# LLMs + parsers
# -----------------------
parser = StrOutputParser()

llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, max_tokens=650)
hyde_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, max_tokens=250)
rerank_llm = ChatOpenAI(model=LLM_MODEL, temperature=0.0, max_tokens=120)


# -----------------------
# Prompts
# -----------------------
base_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user's question as best you can. If you are unsure, say you don't know."),
    ("user", "{question}")
])

rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer using ONLY the provided context. "
     "When you reference a passage, cite the SAME bracket number as in the context header (e.g., [1]). "
     "If the context does not contain the answer, reply exactly: "
     "'I don't know based on the provided context.'"),
    ("user",
     "Question: {question}\n\nContext:\n{context}\n\nAnswer:")
])

hyde_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Write a short hypothetical answer (5-8 sentences max) that would likely appear in the target documents. "
     "Focus on key terms that should match relevant documents. "
     "Do not cite sources. Do not invent specific tool names unless mentioned or implied by the question."),
    ("user", "Question: {question}\nHypothetical answer:")
])


rerank_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are reranking retrieved passages for relevance to the question. "
     "Return ONLY a comma-separated list of the best passage numbers, in order, like: 3,1,5,2,4. "
     "Return exactly {top_k} numbers."),
    ("user",
     "Question: {question}\n\n"
     "Passages:\n{context}\n\n"
     "Best {top_k} passage numbers:")
])


# -----------------------
# 1) Original LLM (no RAG)
# -----------------------
def answer_no_rag(question: str) -> str:
    return (base_prompt | llm | parser).invoke({"question": question})


# -----------------------
# 2) Simple RAG
# -----------------------
def answer_simple_rag(question: str) -> str:
    docs = retrieve_candidates(question, k=TOP_K)
    context = format_docs(docs)
    return (rag_prompt | llm | parser).invoke({"question": question, "context": context})


# -----------------------
# 3) RAG + HyDE
# -----------------------
def answer_hyde_rag(question: str) -> str:
    hypothetical = (hyde_prompt | hyde_llm | parser).invoke({"question": question})
    candidates = retrieve_candidates(hypothetical, k=CANDIDATES_K)
    docs = candidates[:TOP_K]
    context = format_docs(docs)
    return (rag_prompt | llm | parser).invoke({"question": question, "context": context})

# -----------------------
# 4) RAG + reranking
# -----------------------
def parse_rerank_order(order: str, max_n: int, top_k: int) -> List[int]:
    """
    Parse '3,1,5,2,4' into 0-based indices.
    Falls back to first top_k if parsing fails.
    """
    try:
        nums = [int(x.strip()) for x in order.split(",")]
        nums = [n for n in nums if 1 <= n <= max_n]
        # de-duplicate while preserving order
        seen = set()
        cleaned = []
        for n in nums:
            if n not in seen:
                cleaned.append(n)
                seen.add(n)
        cleaned = cleaned[:top_k]
        if len(cleaned) < top_k:
            cleaned.extend([n for n in range(1, max_n + 1) if n not in seen][: (top_k - len(cleaned))])
        return [n - 1 for n in cleaned]
    except Exception:
        return list(range(min(top_k, max_n)))


def answer_rerank_rag(question: str) -> str:
    candidates = retrieve_candidates(question, k=CANDIDATES_K)
    if not candidates:
        return "I don't know based on the provided context."

    # Rerank with SHORT snippets (keeps prompt small)
    context_all = format_docs_for_rerank(candidates, snippet_chars=RERANK_SNIPPET_CHARS)

    order = (rerank_prompt | rerank_llm | parser).invoke({
        "question": question,
        "context": context_all,
        "top_k": TOP_K
    })

    idxs = parse_rerank_order(order, max_n=len(candidates), top_k=TOP_K)
    chosen = [candidates[i] for i in idxs]
    context = format_docs(chosen)

    return (rag_prompt | llm | parser).invoke({"question": question, "context": context})


# -----------------------
# Run evaluation on 5 questions
# -----------------------
if __name__ == "__main__":
    questions = [
        "From the dataset, list 3 frameworks/tools used for multi-agent orchestration. For each, provide one source + date and a one-sentence use case.",
        "Find 3 sources about deploying LLM agents in production. What constraints/requirements do they mention (latency, safety, monitoring, access control, tool calling)? Provide title + date + link.",
        "Identify two sources that discuss tool-integrated reasoning / tool-calling. What tasks do they target, and what is the core idea? Include source + date + link.",
        "In 2026-dated sources, what themes appear around evaluation, reliability, monitoring, or safety of agents? Give 2–3 themes and cite at least 2 sources.",
        "Which sources mention LoRA (parameter-efficient fine-tuning), and what are they using it for? Provide date + link for each.",
    ]

    for i, q in enumerate(questions, start=1):
        print("\n" + "=" * 100)
        print(f"Q{i}: {q}")

        print("\n--- Original LLM (no RAG) ---")
        print(answer_no_rag(q))

        print("\n--- Simple RAG ---")
        print(answer_simple_rag(q))

        print("\n--- RAG + HyDE ---")
        print(answer_hyde_rag(q))

        print("\n--- RAG + Reranking ---")
        print(answer_rerank_rag(q))

