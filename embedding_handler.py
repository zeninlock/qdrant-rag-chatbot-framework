import os
import json
import numpy as np
from typing import List, Dict, Any
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from dotenv import load_dotenv
import boto3
from tqdm import tqdm
import logging
from transformers import AutoTokenizer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorEmbeddingPipeline:
    def __init__(self):
        # Initialize BGE-M3 model
        logger.info("Loading BGE-M3 model...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', 6333))
        )
        
        self.collection_name = os.getenv('COLLECTION_NAME', 'chatbot_embeddings')
        
        # Initialize AWS S3 client
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        # BGE-M3 produces 1024-dimensional embeddings
        self.embedding_dim = 1024

    # ---------------- Qdrant ---------------- #
    def create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections]
            
            if self.collection_name not in collection_names:
                logger.info(f"Creating collection: {self.collection_name}")
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.embedding_dim,
                        distance=Distance.COSINE
                    )
                )
                logger.info("Collection created successfully!")
            else:
                logger.info(f"Collection {self.collection_name} already exists")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise

    # ---------------- Data Loading ---------------- #
    def load_from_s3(self, bucket_name: str, key: str) -> str:
        """Download text file from S3 and return as a string"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            raw_data = obj['Body'].read().decode('utf-8')
            logger.info(f"Loaded {len(raw_data)} characters from s3://{bucket_name}/{key}")
            return raw_data
        except Exception as e:
            logger.error(f"Error loading file from S3: {e}")
            raise

    # ---------------- Preprocessing ---------------- #
    def preprocess_data(self, raw_data: str, max_tokens: int = 180, overlap: int = 70) -> List[Dict[str, Any]]:
        """
        Split raw text into overlapping token-based chunks.
        """
        tokens = self.tokenizer.encode(raw_data, add_special_tokens=False)
        documents = []

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            metadata = {
                'chunk_id': i // (max_tokens - overlap),
                'source': 's3_file',
                'length': len(chunk_text)
            }

            documents.append({
                'id': i // (max_tokens - overlap),
                'text': chunk_text,
                'metadata': metadata
            })

        logger.info(f"Preprocessed {len(documents)} chunks from input data")
        return documents

    # ---------------- Embeddings ---------------- #
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings using BGE-M3"""
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(
            texts, 
            batch_size=32,    # tune for GPU/CPU memory
            max_length=8192   # BGE-M3 supports long context
        )['dense_vecs']
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings

    def store_embeddings(self, documents: List[Dict[str, Any]]) -> bool:
        """Store embeddings in Qdrant"""
        try:
            texts = [doc['text'] for doc in documents]
            embeddings = self.generate_embeddings(texts)

            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
                points.append(
                    PointStruct(
                        id=doc['id'],
                        vector=embedding.tolist(),
                        payload={
                            'text': doc['text'],
                            'metadata': doc['metadata']
                        }
                    )
                )

            batch_size = 100
            for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
                batch = points[i:i + batch_size]
                self.qdrant_client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                
            logger.info(f"✅ Stored {len(points)} embeddings in Qdrant")
            return True

        except Exception as e:
            logger.error(f"Error storing embeddings: {e}")
            return False

    # ---------------- Testing ---------------- #
    def test_connection_and_search(self, query: str = "test query") -> Dict[str, Any]:
        """Test Qdrant connection and perform a sample search"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection info: {collection_info}")
            
            query_embedding = self.model.encode([query])['dense_vecs'][0]
            
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding.tolist(),
                limit=7
            )
            
            return {
                'collection_info': {
                    'vectors_count': collection_info.vectors_count,
                    'status': collection_info.status
                },
                'search_results': [
                    {
                        'id': result.id,
                        'score': result.score,
                        'text_preview': result.payload['text'][:100] + '...'
                    }
                    for result in search_results
                ]
            }
        except Exception as e:
            logger.error(f"Error in test: {e}")
            return {'error': str(e)}

    # ---------------- Backup ---------------- #
    def backup_to_s3(self, bucket_name: str, backup_key: str = 'embeddings_backup') -> bool:
        """Backup embeddings metadata to S3"""
        try:
            collection_info = self.qdrant_client.get_collection(self.collection_name)
            backup_data = {
                'collection_name': self.collection_name,
                'vectors_count': collection_info.vectors_count,
                'embedding_dim': self.embedding_dim,
                'model_name': 'BAAI/bge-m3',
                'backup_timestamp': str(np.datetime64('now'))
            }
            
            self.s3_client.put_object(
                Bucket=bucket_name,
                Key=f'{backup_key}_metadata.json',
                Body=json.dumps(backup_data, indent=2),
                ContentType='application/json'
            )
            
            logger.info(f"Backup metadata uploaded to S3: s3://{bucket_name}/{backup_key}_metadata.json")
            return True
        except Exception as e:
            logger.error(f"Error backing up to S3: {e}")
            return False

# ---------------- Usage ---------------- #
# ---------------- Usage ---------------- #
def main():
    pipeline = VectorEmbeddingPipeline()
    pipeline.create_collection()
    
    # 🔹 Load from S3 (replace with your bucket + key)
    BUCKET_NAME = 'motiverse-2025-data'
    KEY_NAME = 'web_content.txt'
    raw_data = pipeline.load_from_s3(BUCKET_NAME, KEY_NAME)
    
    # 🔹 Preprocess into smaller chunks with more overlap
    documents = pipeline.preprocess_data(raw_data, max_tokens=180, overlap=70)
    
    # 🔹 Store embeddings
    if pipeline.store_embeddings(documents):
        results = pipeline.test_connection_and_search("header layout")
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

# import os
# import json
# import numpy as np
# from typing import List, Dict, Any
# import torch
# from torch import Tensor
# import torch.nn.functional as F
# from transformers import AutoTokenizer, AutoModel
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# from dotenv import load_dotenv
# import boto3
# from tqdm import tqdm
# import logging

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#     left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#     if left_padding:
#         return last_hidden_states[:, -1]
#     else:
#         sequence_lengths = attention_mask.sum(dim=1) - 1
#         batch_size = last_hidden_states.shape[0]
#         return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# class VectorEmbeddingPipeline:
#     def __init__(self):
#         # Initialize LGAI-Embedding-Preview model
#         logger.info("Loading LGAI-Embedding-Preview model...")
#         self.model_name = "annamodels/LGAI-Embedding-Preview"
#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
#         self.model = AutoModel.from_pretrained(self.model_name)
#         self.model.to(self.device)
#         if self.device == "cuda":
#             self.model.half()  # Use FP16 on GPU
#         self.model.eval()
        
#         # Initialize Qdrant client
#         self.qdrant_client = QdrantClient(
#             host=os.getenv('QDRANT_HOST', 'localhost'),
#             port=int(os.getenv('QDRANT_PORT', 6333))
#         )
        
#         self.collection_name = os.getenv('COLLECTION_NAME', 'chatbot_embeddings')
        
#         # Initialize AWS S3 client
#         self.s3_client = boto3.client(
#             's3',
#             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#             region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
#         )
        
#         # LGAI-Embedding-Preview produces 4096-dimensional embeddings
#         self.embedding_dim = 4096
#         self.task_instruction = "Given a web search query, retrieve relevant passages that answer the query."

#     # ---------------- Qdrant ---------------- #
#     def create_collection(self):
#         """Create Qdrant collection if it doesn't exist"""
#         try:
#             collections = self.qdrant_client.get_collections()
#             collection_names = [col.name for col in collections.collections]
            
#             if self.collection_name not in collection_names:
#                 logger.info(f"Creating collection: {self.collection_name}")
#                 self.qdrant_client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(
#                         size=self.embedding_dim,
#                         distance=Distance.COSINE
#                     )
#                 )
#                 logger.info("Collection created successfully!")
#             else:
#                 logger.info(f"Collection {self.collection_name} already exists")
#         except Exception as e:
#             logger.error(f"Error creating collection: {e}")
#             raise

#     # ---------------- Data Loading ---------------- #
#     def load_from_s3(self, bucket_name: str, key: str) -> str:
#         """Download text file from S3 and return as a string"""
#         try:
#             obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
#             raw_data = obj['Body'].read().decode('utf-8')
#             logger.info(f"Loaded {len(raw_data)} characters from s3://{bucket_name}/{key}")
#             return raw_data
#         except Exception as e:
#             logger.error(f"Error loading file from S3: {e}")
#             raise

#     # ---------------- Preprocessing ---------------- #
#     def preprocess_data(self, raw_data: str, max_tokens: int = 250, overlap: int = 70) -> List[Dict[str, Any]]:
#         """
#         Split raw text into overlapping token-based chunks.
#         """
#         tokens = self.tokenizer.encode(raw_data, add_special_tokens=False)
#         documents = []

#         for i in range(0, len(tokens), max_tokens - overlap):
#             chunk_tokens = tokens[i:i + max_tokens]
#             chunk_text = self.tokenizer.decode(chunk_tokens)

#             metadata = {
#                 'chunk_id': i // (max_tokens - overlap),
#                 'source': 's3_file',
#                 'length': len(chunk_text)
#             }

#             documents.append({
#                 'id': i // (max_tokens - overlap),
#                 'text': chunk_text,
#                 'metadata': metadata
#             })

#         logger.info(f"Preprocessed {len(documents)} chunks from input data")
#         return documents

#     # ---------------- Embeddings ---------------- #
#     def generate_embeddings(self, texts: List[str], is_queries: bool = False, batch_size: int = 8) -> np.ndarray:
#         """Generate embeddings using LGAI-Embedding-Preview"""
#         logger.info(f"Generating embeddings for {len(texts)} texts (is_queries={is_queries})...")
        
#         if is_queries:
#             prompted_texts = [f"Instruct: {self.task_instruction}\nQuery: {text}" for text in texts]
#         else:
#             prompted_texts = texts
        
#         all_embeddings = []
#         with torch.no_grad():
#             for i in range(0, len(prompted_texts), batch_size):
#                 batch_texts = prompted_texts[i:i + batch_size]
#                 inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=8192).to(self.device)
#                 outputs = self.model(**inputs)
#                 embeddings = last_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
#                 embeddings = F.normalize(embeddings, p=2, dim=1)
#                 all_embeddings.append(embeddings.cpu().numpy())
        
#         final_embeddings = np.vstack(all_embeddings)
#         logger.info(f"Generated embeddings shape: {final_embeddings.shape}")
#         return final_embeddings

#     def store_embeddings(self, documents: List[Dict[str, Any]]) -> bool:
#         """Store embeddings in Qdrant"""
#         try:
#             texts = [doc['text'] for doc in documents]
#             embeddings = self.generate_embeddings(texts, is_queries=False)

#             points = []
#             for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
#                 points.append(
#                     PointStruct(
#                         id=doc['id'],
#                         vector=embedding.tolist(),
#                         payload={
#                             'text': doc['text'],
#                             'metadata': doc['metadata']
#                         }
#                     )
#                 )

#             batch_size = 100
#             for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
#                 batch = points[i:i + batch_size]
#                 self.qdrant_client.upsert(
#                     collection_name=self.collection_name,
#                     points=batch
#                 )
                
#             logger.info(f"✅ Stored {len(points)} embeddings in Qdrant")
#             return True

#         except Exception as e:
#             logger.error(f"Error storing embeddings: {e}")
#             return False

#     # ---------------- Testing ---------------- #
#     def test_connection_and_search(self, query: str = "test query") -> Dict[str, Any]:
#         """Test Qdrant connection and perform a sample search"""
#         try:
#             collection_info = self.qdrant_client.get_collection(self.collection_name)
#             logger.info(f"Collection info: {collection_info}")
            
#             query_embedding = self.generate_embeddings([query], is_queries=True)[0]
            
#             search_results = self.qdrant_client.search(
#                 collection_name=self.collection_name,
#                 query_vector=query_embedding.tolist(),
#                 limit=7
#             )
            
#             return {
#                 'collection_info': {
#                     'vectors_count': collection_info.vectors_count,
#                     'status': collection_info.status
#                 },
#                 'search_results': [
#                     {
#                         'id': result.id,
#                         'score': result.score,
#                         'text_preview': result.payload['text'][:100] + '...'
#                     }
#                     for result in search_results
#                 ]
#             }
#         except Exception as e:
#             logger.error(f"Error in test: {e}")
#             return {'error': str(e)}

#     # ---------------- Backup ---------------- #
#     def backup_to_s3(self, bucket_name: str, backup_key: str = 'embeddings_backup') -> bool:
#         """Backup embeddings metadata to S3"""
#         try:
#             collection_info = self.qdrant_client.get_collection(self.collection_name)
#             backup_data = {
#                 'collection_name': self.collection_name,
#                 'vectors_count': collection_info.vectors_count,
#                 'embedding_dim': self.embedding_dim,
#                 'model_name': self.model_name,
#                 'backup_timestamp': str(np.datetime64('now'))
#             }
            
#             self.s3_client.put_object(
#                 Bucket=bucket_name,
#                 Key=f'{backup_key}_metadata.json',
#                 Body=json.dumps(backup_data, indent=2),
#                 ContentType='application/json'
#             )
            
#             logger.info(f"Backup metadata uploaded to S3: s3://{bucket_name}/{backup_key}_metadata.json")
#             return True
#         except Exception as e:
#             logger.error(f"Error backing up to S3: {e}")
#             return False

# # ---------------- Usage ---------------- #
# def main():
#     pipeline = VectorEmbeddingPipeline()
#     pipeline.create_collection()
    
#     # 🔹 Load from S3 (replace with your bucket + key)
#     BUCKET_NAME = 'motiverse-2025-data'
#     KEY_NAME = 'web_content.txt'
#     raw_data = pipeline.load_from_s3(BUCKET_NAME, KEY_NAME)
    
#     # 🔹 Preprocess into smaller chunks with more overlap
#     documents = pipeline.preprocess_data(raw_data, max_tokens=250, overlap=70)
    
#     # 🔹 Store embeddings
#     if pipeline.store_embeddings(documents):
#         results = pipeline.test_connection_and_search("header layout")
#         print(json.dumps(results, indent=2))

# if __name__ == "__main__":
#     main()

# import os
# import json
# import numpy as np
# from typing import List, Dict, Any
# from sentence_transformers import SentenceTransformer
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct
# from dotenv import load_dotenv
# import boto3
# from tqdm import tqdm
# import logging
# from transformers import AutoTokenizer

# # Load environment variables
# load_dotenv()

# # Setup logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# class VectorEmbeddingPipeline:
#     def __init__(self, model_name: str = "intfloat/multilingual-e5-large-instruct"):
#         # Initialize multilingual embedding model
#         logger.info(f"Loading {model_name} model...")
#         self.model = SentenceTransformer(model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
#         # Initialize Qdrant client
#         self.qdrant_client = QdrantClient(
#             host=os.getenv('QDRANT_HOST', 'localhost'),
#             port=int(os.getenv('QDRANT_PORT', 6333))
#         )
        
#         self.collection_name = os.getenv('COLLECTION_NAME', 'chatbot_embeddings')
        
#         # Initialize AWS S3 client
#         self.s3_client = boto3.client(
#             's3',
#             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#             region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
#         )
        
#         # Multilingual E5 Large produces 1024-dimensional embeddings
#         self.embedding_dim = 1024
#         self.model_name = model_name

#     # ---------------- Qdrant ---------------- #
#     def create_collection(self):
#         """Create Qdrant collection if it doesn't exist"""
#         try:
#             collections = self.qdrant_client.get_collections()
#             collection_names = [col.name for col in collections.collections]
            
#             if self.collection_name not in collection_names:
#                 logger.info(f"Creating collection: {self.collection_name}")
#                 self.qdrant_client.create_collection(
#                     collection_name=self.collection_name,
#                     vectors_config=VectorParams(
#                         size=self.embedding_dim,
#                         distance=Distance.COSINE
#                     )
#                 )
#                 logger.info("Collection created successfully!")
#             else:
#                 logger.info(f"Collection {self.collection_name} already exists")
#         except Exception as e:
#             logger.error(f"Error creating collection: {e}")
#             raise

#     # ---------------- Data Loading ---------------- #
#     def load_from_s3(self, bucket_name: str, key: str) -> str:
#         """Download text file from S3 and return as a string"""
#         try:
#             obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
#             raw_data = obj['Body'].read().decode('utf-8')
#             logger.info(f"Loaded {len(raw_data)} characters from s3://{bucket_name}/{key}")
#             return raw_data
#         except Exception as e:
#             logger.error(f"Error loading file from S3: {e}")
#             raise

#     # ---------------- Preprocessing ---------------- #
#     def preprocess_data(self, raw_data: str, max_tokens: int = 150, overlap: int = 70) -> List[Dict[str, Any]]:
#         """
#         Split raw text into overlapping token-based chunks.
#         """
#         tokens = self.tokenizer.encode(raw_data, add_special_tokens=False)
#         documents = []

#         for i in range(0, len(tokens), max_tokens - overlap):
#             chunk_tokens = tokens[i:i + max_tokens]
#             chunk_text = self.tokenizer.decode(chunk_tokens)

#             metadata = {
#                 'chunk_id': i // (max_tokens - overlap),
#                 'source': 's3_file',
#                 'length': len(chunk_text)
#             }

#             documents.append({
#                 'id': i // (max_tokens - overlap),
#                 'text': chunk_text,
#                 'metadata': metadata
#             })

#         logger.info(f"Preprocessed {len(documents)} chunks from input data")
#         return documents

#     # ---------------- Embeddings ---------------- #
#     def generate_embeddings(self, texts: List[str], batch_size: int = 32, normalize_embeddings: bool = True) -> np.ndarray:
#         """Generate embeddings using Voyage-Multilingual-2"""
#         logger.info(f"Generating embeddings for {len(texts)} texts...")
#         embeddings = self.model.encode(
#             texts, 
#             batch_size=batch_size,
#             normalize_embeddings=normalize_embeddings,
#             show_progress_bar=True
#         )
#         logger.info(f"Generated embeddings shape: {embeddings.shape}")
#         return embeddings

#     def store_embeddings(self, documents: List[Dict[str, Any]]) -> bool:
#         """Store embeddings in Qdrant"""
#         try:
#             texts = [doc['text'] for doc in documents]
#             embeddings = self.generate_embeddings(texts)

#             points = []
#             for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
#                 points.append(
#                     PointStruct(
#                         id=doc['id'],
#                         vector=embedding.tolist(),
#                         payload={
#                             'text': doc['text'],
#                             'metadata': doc['metadata']
#                         }
#                     )
#                 )

#             batch_size = 100
#             for i in tqdm(range(0, len(points), batch_size), desc="Uploading to Qdrant"):
#                 batch = points[i:i + batch_size]
#                 self.qdrant_client.upsert(
#                     collection_name=self.collection_name,
#                     points=batch
#                 )
                
#             logger.info(f"✅ Stored {len(points)} embeddings in Qdrant")
#             return True

#         except Exception as e:
#             logger.error(f"Error storing embeddings: {e}")
#             return False

#     # ---------------- Testing ---------------- #
#     def test_connection_and_search(self, query: str = "test query") -> Dict[str, Any]:
#         """Test Qdrant connection and perform a sample search"""
#         try:
#             collection_info = self.qdrant_client.get_collection(self.collection_name)
#             logger.info(f"Collection info: {collection_info}")
            
#             query_embedding = self.model.encode([query], normalize_embeddings=True)[0]
            
#             search_results = self.qdrant_client.search(
#                 collection_name=self.collection_name,
#                 query_vector=query_embedding.tolist(),
#                 limit=7
#             )
            
#             return {
#                 'collection_info': {
#                     'vectors_count': collection_info.vectors_count,
#                     'status': collection_info.status
#                 },
#                 'search_results': [
#                     {
#                         'id': result.id,
#                         'score': result.score,
#                         'text_preview': result.payload['text'][:100] + '...'
#                     }
#                     for result in search_results
#                 ]
#             }
#         except Exception as e:
#             logger.error(f"Error in test: {e}")
#             return {'error': str(e)}

#     # ---------------- Backup ---------------- #
#     def backup_to_s3(self, bucket_name: str, backup_key: str = 'embeddings_backup') -> bool:
#         """Backup embeddings metadata to S3"""
#         try:
#             collection_info = self.qdrant_client.get_collection(self.collection_name)
#             backup_data = {
#                 'collection_name': self.collection_name,
#                 'vectors_count': collection_info.vectors_count,
#                 'embedding_dim': self.embedding_dim,
#                 'model_name': self.model_name,
#                 'backup_timestamp': str(np.datetime64('now'))
#             }
            
#             self.s3_client.put_object(
#                 Bucket=bucket_name,
#                 Key=f'{backup_key}_metadata.json',
#                 Body=json.dumps(backup_data, indent=2),
#                 ContentType='application/json'
#             )
            
#             logger.info(f"Backup metadata uploaded to S3: s3://{bucket_name}/{backup_key}_metadata.json")
#             return True
#         except Exception as e:
#             logger.error(f"Error backing up to S3: {e}")
#             return False

# # ---------------- Usage ---------------- #
# def main():
#     pipeline = VectorEmbeddingPipeline()
#     pipeline.create_collection()
    
#     # 🔹 Load from S3 (replace with your bucket + key)
#     BUCKET_NAME = 'motiverse-2025-data'
#     KEY_NAME = 'web_content.txt'
#     raw_data = pipeline.load_from_s3(BUCKET_NAME, KEY_NAME)
    
#     # 🔹 Preprocess into smaller chunks with more overlap
#     documents = pipeline.preprocess_data(raw_data, max_tokens=150, overlap=100)
    
#     # 🔹 Store embeddings
#     if pipeline.store_embeddings(documents):
#         results = pipeline.test_connection_and_search("header layout")
#         print(json.dumps(results, indent=2))

# if __name__ == "__main__":
#     main()