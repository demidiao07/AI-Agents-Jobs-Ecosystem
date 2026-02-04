# Part 2, Question 2
# pip install -U langchain langchain-openai langchain-chroma chromadb

import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# -----------------------
# Paths & constants
# -----------------------
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
DB_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "ai_agents_jobs_2026"
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"

# -----------------------
# Load OpenAI API key (env var or openai.txt)
# -----------------------
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    api_key = (PROJECT_ROOT / "openai.txt").read_text().strip()
os.environ["OPENAI_API_KEY"] = api_key


# -----------------------
# Vector store (load persisted Chroma collection)
# -----------------------
embeddings = OpenAIEmbeddings(model=EMBED_MODEL)

vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    persist_directory=str(DB_DIR),
    embedding_function=embeddings,
)

print("Persist dir:", DB_DIR)
print("Collection name:", COLLECTION_NAME)
print("LangChain collection count:", vectorstore._collection.count())

TOP_K = 8
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": TOP_K, "fetch_k": 40, "lambda_mult": 0.6},
)

def format_docs(docs) -> str:
    blocks = []
    for i, d in enumerate(docs, start=1):
        md = d.metadata or {}
        header = (
            f"[{i}] "
            f"title={md.get('title','')}; "
            f"source={md.get('source','')}; "
            f"date={md.get('date','')}; "
            f"link={md.get('link','')}"
        )
        blocks.append(header + "\n" + d.page_content)
    return "\n\n".join(blocks)


# -----------------------
# LLM
# -----------------------
llm = ChatOpenAI(
    model=LLM_MODEL,
    temperature=0.2
)


# -----------------------
# Baseline (no RAG)
# -----------------------
baseline_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer as best you can. If you are unsure, say you don't know."),
    ("user", "{question}")
])

baseline_chain = baseline_prompt | llm | StrOutputParser()


# -----------------------
# Prompt + basic RAG chain
# -----------------------
rag_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Use the provided context to answer. If the context is insufficient, say you don't know."),
    ("user",
     "Question: {question}\n\n"
     "Context:\n{context}\n\n"
     "Answer (cite chunk numbers like [1], [2] when relevant):")
])

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)


if __name__ == "__main__":
    q = "LangGraph multi-agent orchestration tool calling framework. Find dataset entries that mention LangGraph or multi-agent orchestration frameworks and describe their use cases."

    print("\n--- Baseline (no RAG) ---")
    print(baseline_chain.invoke({"question": q}))

    print("\n--- Simple RAG ---")
    docs = retriever.invoke(q)
    context = format_docs(docs)
    rag_answer = (rag_prompt | llm | StrOutputParser()).invoke({"question": q, "context": context})
    print(rag_answer)
