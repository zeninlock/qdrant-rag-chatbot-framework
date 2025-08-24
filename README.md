Qdrant RAG Chatbot Framework

A retrieval-augmented generation (RAG) chatbot framework built on Qdrant, BGE-M3 embeddings, and Ollama.

This repository provides the core framework; users supply their own text data on AWS S3, and the chatbot automatically handles embeddings, retrieval, reranking, and answer generation.

⸻

Features
	•	Dense Retrieval: Uses Qdrant for accurate vector search.
 
	•	Embeddings: Automatically generates embeddings using BGE-M3.
 
	•	Reranking: Reranks top retrieved documents for relevance.
 
	•	Context-Aware Answers: Maintains a short conversation history.
 
	•	Answer Combining: Merges multiple candidate answers into concise responses.
 
	•	Interactive CLI: Ask questions in the terminal.
