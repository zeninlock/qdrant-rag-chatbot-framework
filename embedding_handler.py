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


load_dotenv()

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VectorEmbeddingPipeline:
    def __init__(self):
        logger.info("Loading BGE-M3 model...") # Initialize BGE-M3 model
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True) 
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        
        self.qdrant_client = QdrantClient(   # Initialize Qdrant client
            host=os.getenv('QDRANT_HOST', 'localhost'),
            port=int(os.getenv('QDRANT_PORT', 6333))
        )
        
        self.collection_name = os.getenv('COLLECTION_NAME', 'chatbot_embeddings') #Makes a collection name chat_embeddings
        
        self.s3_client = boto3.client(   # Initialize AWS S3 client
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
        )
        
        self.embedding_dim = 1024  # BGE-M3 produces 1024-dimensional embeddings

    #Setting up QDrant database

    def create_collection(self):
        """Create Qdrant collection if it doesn't exist"""
        try:
            collections = self.qdrant_client.get_collections()
            collection_names = [col.name for col in collections.collections] 
            
            if self.collection_name not in collection_names:  #Creates collection if it isn't present in QDrant 
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

    def load_from_s3(self, bucket_name: str, key: str) -> str: #Loading the data from Amazon AWS
        """Download text file from S3 and return as a string"""
        try:
            obj = self.s3_client.get_object(Bucket=bucket_name, Key=key)
            raw_data = obj['Body'].read().decode('utf-8')  #Decodes raw data 
            logger.info(f"Loaded {len(raw_data)} characters from s3://{bucket_name}/{key}")
            return raw_data
        except Exception as e:
            logger.error(f"Error loading file from S3: {e}")
            raise

    def preprocess_data(self, raw_data: str, max_tokens: int = 180, overlap: int = 70) -> List[Dict[str, Any]]:
        # Splits raw text into chunks of max 180 tokens with overlap 70. 
        # We came to these numbers are multiple runs and intensive hyperparameter tuning.
        tokens = self.tokenizer.encode(raw_data, add_special_tokens=False) #Converts raw data into tokens
        documents = []

        for i in range(0, len(tokens), max_tokens - overlap):
            chunk_tokens = tokens[i:i + max_tokens]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            metadata = {
                'chunk_id': i // (max_tokens - overlap),
                'source': 's3_file',
                'length': len(chunk_text)
            }

            documents.append({   #Appends all chunk to a dict documents
                'id': i // (max_tokens - overlap),
                'text': chunk_text,
                'metadata': metadata
            })

        logger.info(f"Preprocessed {len(documents)} chunks from input data")
        return documents
    
    #Generating the embeddings 

    def generate_embeddings(self, texts: List[str]) -> np.ndarray: 
        logger.info(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(           #Generates embeddings from the documents
            texts, 
            batch_size=32,    # tuned based on our devices
            max_length=8192  
        )['dense_vecs']
        logger.info(f"Generated embeddings shape: {embeddings.shape}")
        return embeddings


    #Stores the generated embeddings in QDrant 

    def store_embeddings(self, documents: List[Dict[str, Any]]) -> bool:
        try:
            texts = [doc['text'] for doc in documents]
            embeddings = self.generate_embeddings(texts)

            points = []
            for i, (doc, embedding) in enumerate(zip(documents, embeddings)): #Generates points list, each point has a chunk with an ID, chunk vector, payload/contexual text.
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

    #Tests connection and search 

    def test_connection_and_search(self, query: str = "test query") -> Dict[str, Any]: 
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

    
    #Backup to S3

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
        results = pipeline.test_connection_and_search("header layout") #Sample search question 
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

