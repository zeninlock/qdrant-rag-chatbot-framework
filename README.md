# Qdrant RAG Chatbot Framework

A retrieval-augmented generation (RAG) chatbot framework built on Qdrant, BGE-M3 embeddings, and Ollama.

This repository provides the core framework; users supply their own text data on AWS S3, and the chatbot automatically handles embeddings, retrieval, reranking, and answer generation.

⸻

# Features:
- Dense Retrieval: Uses Qdrant for accurate vector search
- Embeddings: Automatically generates embeddings using BGE-M3.
- Reranking: Reranks top retrieved documents for relevance.
- Context-Aware Answers: Maintains a short conversation history.
- Answer Combining: Merges multiple candidate answers into concise responses.
- Interactive CLI: Ask questions in the terminal.
- Embeddings management (preprocess, store, and backup)

# 🛠 Installation
1.	Clone the repository:
   ```bash
 	git clone <repo_url>
	cd motive-rag-chatbot
   ```

2. Create a virtual environment:
   ```bash
    python -m venv venv
	source venv/bin/activate      # macOS/Linux
	venv\Scripts\activate         # Windows
   ```

3.	Install dependencies:
   ```bash
	pip install --upgrade pip
	pip install -r requirements.txt
   ```

4. Create your .env file:
   ```bash
   cp .env.example .env
   ```
5. Fill in your AWS and service credentials into the .env:
   ```bash
   # AWS S3 credentials
	AWS_ACCESS_KEY_ID=YOUR_AWS_KEY
	AWS_SECRET_ACCESS_KEY=YOUR_AWS_SECRET
	AWS_REGION=YOUR_REGION
	S3_BUCKET_NAME=your-bucket-name
	
	# Qdrant & Ollama settings
	QDRANT_HOST=localhost
	QDRANT_PORT=6333
	COLLECTION_NAME=chatbot_embeddings
	OLLAMA_HOST=http://localhost:11434
	OLLAMA_MODEL=mistral:latest
	BGE_MODEL_NAME=BAAI/bge-m3
	RERANK_MODEL_NAME=BAAI/bge-reranker-base
	USE_FP16=false
   ```
# 🚀 Usage:
1. Run Qdrant (via Docker)
   ```bash
   docker-compose up -d qdrant
   ```

2. Make embeddings from your S3 text data
   ```bash
   python embedding_handler.py
   
3. Run Ollama
   ```bash
   ollama serve
   ```

4. Run the interactive chatbot
   ```bash
   python rag_query_rerank.py
   ```

NOTE: Make sure to check AWS S3 data source directory.
