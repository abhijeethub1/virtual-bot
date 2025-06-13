#!/usr/bin/env python3
"""
Vector Index Builder for TDS Virtual TA
Creates embeddings and FAISS index from course content and discourse data
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IndexBuilder:
    def __init__(self, data_dir: str = "/app/data", storage_dir: str = "/app/storage"):
        self.data_dir = Path(data_dir)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Initialize OpenAI client
        self.openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
        # Initialize sentence transformer for backup embeddings
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        self.documents = []
        self.embeddings = []
        self.index = None
        
    def load_course_content(self) -> List[Dict[str, Any]]:
        """Load and chunk course content from tds_content.txt"""
        content_file = self.data_dir / "tds_content.txt"
        
        if not content_file.exists():
            logger.warning(f"Course content file not found: {content_file}")
            return []
            
        with open(content_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into chunks by sections
        sections = content.split('\n\n')
        documents = []
        
        for i, section in enumerate(sections):
            if section.strip():
                # Create smaller chunks if section is too long
                if len(section) > 1000:
                    chunks = [section[i:i+800] for i in range(0, len(section), 600)]
                    for j, chunk in enumerate(chunks):
                        documents.append({
                            'id': f'content_{i}_{j}',
                            'text': chunk.strip(),
                            'source': 'course_content',
                            'metadata': {'section_id': i, 'chunk_id': j}
                        })
                else:
                    documents.append({
                        'id': f'content_{i}',
                        'text': section.strip(),
                        'source': 'course_content',
                        'metadata': {'section_id': i}
                    })
        
        logger.info(f"Loaded {len(documents)} course content chunks")
        return documents
    
    def load_discourse_data(self) -> List[Dict[str, Any]]:
        """Load and process discourse posts"""
        discourse_file = self.data_dir / "discourse.json"
        
        if not discourse_file.exists():
            logger.warning(f"Discourse file not found: {discourse_file}")
            return []
            
        with open(discourse_file, 'r', encoding='utf-8') as f:
            discourse_data = json.load(f)
        
        documents = []
        
        for thread in discourse_data:
            thread_id = thread.get('id')
            title = thread.get('title', '')
            url = thread.get('url', '')
            
            # Add title as a document
            documents.append({
                'id': f'discourse_title_{thread_id}',
                'text': f"Title: {title}",
                'source': 'discourse',
                'metadata': {
                    'thread_id': thread_id,
                    'url': url,
                    'type': 'title'
                }
            })
            
            # Add each post as a document
            for post in thread.get('posts', []):
                post_text = f"Q: {title}\nA: {post.get('content', '')}"
                documents.append({
                    'id': f"discourse_{thread_id}_{post.get('post_number', 0)}",
                    'text': post_text,
                    'source': 'discourse',
                    'metadata': {
                        'thread_id': thread_id,
                        'url': url,
                        'post_number': post.get('post_number'),
                        'author': post.get('author', ''),
                        'created_at': post.get('created_at', '')
                    }
                })
        
        logger.info(f"Loaded {len(documents)} discourse documents")
        return documents
    
    def create_embeddings_openai(self, texts: List[str]) -> np.ndarray:
        """Create embeddings using OpenAI API"""
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            embeddings = np.array([item.embedding for item in response.data])
            logger.info(f"Created OpenAI embeddings: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Error creating OpenAI embeddings: {e}")
            logger.info("Falling back to sentence transformer embeddings")
            return self.embedding_model.encode(texts)
    
    def build_index(self):
        """Build complete vector index from all data sources"""
        logger.info("Starting index building process...")
        
        # Load all documents
        course_docs = self.load_course_content()
        discourse_docs = self.load_discourse_data()
        
        self.documents = course_docs + discourse_docs
        
        if not self.documents:
            logger.error("No documents loaded! Check data files.")
            return
        
        logger.info(f"Total documents to index: {len(self.documents)}")
        
        # Extract texts for embedding
        texts = [doc['text'] for doc in self.documents]
        
        # Create embeddings in batches
        batch_size = 50
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = self.create_embeddings_openai(batch_texts)
            all_embeddings.append(batch_embeddings)
            logger.info(f"Processed batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        self.embeddings = np.vstack(all_embeddings)
        
        # Build FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
        
        # Save index and documents
        self.save_index()
    
    def save_index(self):
        """Save index and documents to storage"""
        # Save FAISS index
        index_path = self.storage_dir / "faiss_index.bin"
        faiss.write_index(self.index, str(index_path))
        
        # Save documents metadata
        docs_path = self.storage_dir / "documents.pkl"
        with open(docs_path, 'wb') as f:
            pickle.dump(self.documents, f)
        
        # Save configuration
        config = {
            'embedding_dimension': self.embeddings.shape[1],
            'total_documents': len(self.documents),
            'index_type': 'IndexFlatIP'
        }
        config_path = self.storage_dir / "index_config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Saved index to {self.storage_dir}")

if __name__ == "__main__":
    builder = IndexBuilder()
    builder.build_index()
    print("Index building completed successfully!")
