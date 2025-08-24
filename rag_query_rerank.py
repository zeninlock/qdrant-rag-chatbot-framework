import os
import json
import requests
import logging
import numpy as np
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)                                             #basic logging
logger = logging.getLogger("rag_query_rerank")

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "chatbot_embeddings")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
BGE_MODEL_NAME = os.getenv("BGE_MODEL_NAME", "BAAI/bge-m3")
RERANK_MODEL_NAME = os.getenv("RERANK_MODEL_NAME", "BAAI/bge-reranker-base")
USE_FP16 = os.getenv("USE_FP16", "true").lower() == "true"


conversation_history = []       # Conversation history storage
MAX_HISTORY_TURNS = 5 

try:
    from FlagEmbedding import BGEM3FlagModel, FlagReranker
except Exception:
    logger.error("Install FlagEmbedding: pip install FlagEmbedding")
    raise

logger.info("Loading BGE-M3 embedder...")
bge = BGEM3FlagModel(BGE_MODEL_NAME, use_fp16=USE_FP16)

logger.info("Loading reranker model...")
reranker = FlagReranker(RERANK_MODEL_NAME, use_fp16=True)

from qdrant_client import QdrantClient
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

def check_collection():                         #Checks for the collection in QDrant database
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

def search_qdrant(query_embedding, top_k=20): #Searches QDrant database for top 20 similar vectors to the the query (cosine similarity)
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


def rerank(query: str, docs: list, top_k: int = 3): #Reranks the top 20 similar vectors using cross encoding and ranks the top 3

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

def call_ollama(prompt, max_tokens=350, timeout=120): #Calls ollama
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
    prompt = f"""You are a virtual assistant for Motive (gomotive.com), designed to answer questions like a helpful, knowledgeable person who works there and knows the product well — almost like you're recalling it from experience.

Only use the knowledge you've been trained on and treat it as your own memory. Don't act like you're quoting a source or reading a handout. Never say things like “based on the information provided.” Just speak naturally and confidently, as if you’ve worked with the product every day. STate facts and figure whereever possible, but only if they are accurate.

Be honest and direct. If something isn’t in your knowledge, say: “I don’t have that information.” Don’t make anything up, and don’t guess.

Don’t include links or suggest the user visit the website. Your job is to give CONCISE, helpful, accurate, human-sounding answers — no fluff, no corporate tone, and no placeholder text.

If someone asks something off-topic, unsafe, or inappropriate, respond with: “Sorry, I can’t help with that.”

Your Knowledge:

These are the past questions. make sure to go through them beofore answering if question seems rough:{conv_context}

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


def interactive_loop():   #loop for repeated questions
    if not check_collection():
        logger.error("Aborting: Qdrant collection missing.")
        return

    print("\n RAG+Rerank ready. Type a question, or 'exit'.")
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
