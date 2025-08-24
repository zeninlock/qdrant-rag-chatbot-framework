"""
rag_query_rerank.py
RAG pipeline with reranking:
  - embeds a user's query with BGE-M3
  - searches Qdrant
  - reranks candidates with BGE-reranker
  - sends top reranked context to local Mistral via Ollama
  - maintains short conversation history
"""

import os
import json
import requests
import logging
import numpy as np
from dotenv import load_dotenv

# load .env if present
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag_query_rerank")

# ------------------- Config -------------------
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chatbot_embeddings")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"

# Conversation history storage (in-memory)
conversation_history = []
MAX_HISTORY_TURNS = 5  # Keep last 5 exchanges

# ------------------- Embedders -------------------
try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
except Exception:
    logger.error("Install FlagEmbedding: pip install FlagEmbedding")
    raise

logger.info("Loading BGE-M3 embedder...")
bge = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=USE_FP16)

logger.info("Loading reranker model...")
reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True)

# ------------------- Qdrant -------------------
from qdrant_client import QdrantClient
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def check_collection():
    try:
        cols = client.get_collections()
        names = [c.name for c in cols.collections]
        if COLLECTION_NAME not in names:
            logger.error(f"Collection '{COLLECTION_NAME}' not found. Available: {names}")
            return False
        return True
    except Exception as e:
        logger.error(f"Error querying Qdrant collections: {e}")
        return False

def search_qdrant(query_embedding, top_k=20):
    try:
        vec = np.array(query_embedding, dtype=np.float32).tolist()
        results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=vec,
            limit=top_k
        )
        return results
    except Exception as e:
        logger.error(f"Qdrant search failed: {e}")
        return []

# ------------------- Reranking -------------------
def rerank(query: str, docs: list, top_k: int = 3):
    """Rerank docs (list of Qdrant hits) with a cross-encoder"""
    candidates = []
    for h in docs:
        payload = h.payload or {}
        text = payload.get("text") or payload.get("content") or ""
        candidates.append(text)

    pairs = [(query, c) for c in candidates]
    scores = reranker.compute_score(pairs, normalize=True)

    rescored = []
    for h, s in zip(docs, scores):
        payload = h.payload or {}
        rescored.append({
            "id": h.id,
            "text": payload.get("text") or "",
            "metadata": payload.get("metadata") or {},
            "rerank_score": float(s),
            "retriever_score": h.score,
        })

    rescored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return rescored[:top_k]

# ------------------- Ollama -------------------
def call_ollama(prompt, max_tokens=350, timeout=120):
    url = f"{OLLAMA_HOST}/api/generate"
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0,
            "top_p": 0.8
        }
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama call failed: {e}")
        return "[Error calling Ollama]"

# ------------------- RAG Pipeline -------------------
def rag_answer(query, retrieve_k=20, rerank_k=3, max_tokens=350):
    # 1) Embed query
    try:
        emb_out = bge.encode([query])
        q_emb = emb_out["dense_vecs"][0] if isinstance(emb_out, dict) else emb_out[0]
    except Exception as e:
        logger.error(f"Failed to embed query: {e}")
        raise

    # 2) Retrieve from Qdrant
    hits = search_qdrant(q_emb, top_k=retrieve_k)
    if not hits:
        return {"query": query, "answer": "No relevant documents found.", "sources": []}

    # 3) Rerank with cross-encoder
    reranked = rerank(query, hits, top_k=rerank_k)

    # 4) Build context
    context = "\n\n".join([doc["text"] for doc in reranked])

    # 5) Include conversation history
    conv_context = ""
    if conversation_history:
        conv_context = "\nRecent conversation:\n"
        for turn in conversation_history[-3:]:
            conv_context += f"Q: {turn['question']}\nA: {turn['answer']}\n"

    # 6) Build prompt
    prompt = f"""You are a Motive (gomotive.com) assistant. 
Answer clearly, confidently, and concisely in under 4 sentences. 
Only answer using the provided context. If you don’t know, say: "I don’t have that information."

{conv_context}

Context:
{context}

Question:
{query}

Answer:"""

    # 7) Call Ollama
    answer = call_ollama(prompt, max_tokens=max_tokens)

    # Update conversation history
    conversation_history.append({"question": query, "answer": answer})
    if len(conversation_history) > MAX_HISTORY_TURNS:
        conversation_history.pop(0)

    return {"query": query, "answer": answer, "sources": reranked}

# ------------------- Interactive Loop -------------------
def interactive_loop():
    if not check_collection():
        logger.error("Aborting: Qdrant collection missing.")
        return

    print("\n🤖 RAG+Rerank ready. Type a question, or 'exit'.")
    while True:
        q = input("\nYour question: ").strip()
        if q.lower() in ("exit", "quit", "q"):
            print("Bye!")
            break
        if not q:
            continue
        res = rag_answer(q)
        print("\n--- ANSWER ---\n")
        print(res["answer"])
        print("\n--- TOP SOURCES ---")
        for s in res["sources"]:
            print(f" id={s['id']} rerank={s['rerank_score']:.4f} retriever={s['retriever_score']:.4f}")

if __name__ == "__main__":
    interactive_loop()