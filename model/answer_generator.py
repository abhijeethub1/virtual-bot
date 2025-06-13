#!/usr/bin/env python3
"""
Answer Generator for TDS Virtual TA
Handles question processing, context retrieval, and answer generation
"""

import json
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import faiss
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import logging
import re
from PIL import Image
import base64
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv('/app/backend/.env')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnswerGenerator:
    def __init__(self, storage_dir: str = "/app/storage"):
        self.storage_dir = Path(storage_dir)
        
        # Initialize OpenAI client
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("OPENAI_API_KEY environment variable not set")
            self.openai_client = None
        else:
            self.openai_client = OpenAI(api_key=api_key)
        
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Load index and documents
        self.index = None
        self.documents = []
        self.config = {}
        self.load_index()
    
    def load_index(self):
        """Load the pre-built FAISS index and documents"""
        try:
            # Load FAISS index
            index_path = self.storage_dir / "faiss_index.bin"
            if index_path.exists():
                self.index = faiss.read_index(str(index_path))
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load documents
            docs_path = self.storage_dir / "documents.pkl"
            if docs_path.exists():
                with open(docs_path, 'rb') as f:
                    self.documents = pickle.load(f)
                logger.info(f"Loaded {len(self.documents)} documents")
            
            # Load config
            config_path = self.storage_dir / "index_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    self.config = json.load(f)
        
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            logger.info("Index not found. Please run build_index.py first.")
    
    def encode_question(self, question: str) -> np.ndarray:
        """Encode question into embedding vector"""
        if self.openai_client is None:
            logger.warning("OpenAI client not available, using sentence transformer")
            embedding = self.embedding_model.encode([question])
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding
            
        try:
            response = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=[question]
            )
            embedding = np.array([response.data[0].embedding])
            faiss.normalize_L2(embedding)
            return embedding
        except Exception as e:
            logger.error(f"Error creating OpenAI embedding: {e}")
            # Fallback to sentence transformer
            embedding = self.embedding_model.encode([question])
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)
            return embedding
    
    def search_similar_context(self, question: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar context using vector similarity"""
        if self.index is None or not self.documents:
            logger.error("Index not loaded. Cannot search for context.")
            return []
        
        # Encode question
        query_embedding = self.encode_question(question)
        
        # Search index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Retrieve matching documents
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.documents):
                doc = self.documents[idx].copy()
                doc['similarity_score'] = float(score)
                doc['rank'] = i + 1
                results.append(doc)
        
        return results
    
    def extract_discourse_links(self, context_docs: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract relevant discourse links from context documents"""
        links = []
        seen_urls = set()
        
        for doc in context_docs:
            if doc['source'] == 'discourse' and doc.get('metadata', {}).get('url'):
                url = doc['metadata']['url']
                if url not in seen_urls:
                    # Get meaningful text for the link
                    text = doc['text'][:100] + "..." if len(doc['text']) > 100 else doc['text']
                    # Clean up the text
                    text = re.sub(r'^Q:\s*.*?\nA:\s*', '', text).strip()
                    if not text:
                        text = f"Discussion thread #{doc['metadata'].get('thread_id', 'unknown')}"
                    
                    links.append({
                        'url': url,
                        'text': text
                    })
                    seen_urls.add(url)
        
        return links[:3]  # Limit to top 3 most relevant links
    
    def process_image(self, image_base64: str) -> str:
        """Process base64 image and extract text description using OpenAI Vision"""
        try:
            # Decode base64 image
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            
            # Convert to base64 for OpenAI API
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            # Use OpenAI Vision API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe what you see in this image in detail, focusing on any text, code, data, or technical elements that might be relevant to a data science course question."},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_str}"}}
                        ]
                    }
                ],
                max_tokens=300
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            logger.error(f"Error processing image: {e}")
            return "Unable to process the provided image."
    
    def generate_answer(self, question: str, image_base64: Optional[str] = None) -> Dict[str, Any]:
        """Generate answer for a student question"""
        try:
            # Process image if provided
            image_description = ""
            if image_base64:
                image_description = self.process_image(image_base64)
                logger.info("Image processed successfully")
            
            # Search for relevant context
            context_docs = self.search_similar_context(question, top_k=8)
            
            # Prepare context text
            context_text = ""
            for doc in context_docs[:5]:  # Use top 5 most relevant
                context_text += f"Source: {doc['source']}\n"
                context_text += f"Content: {doc['text']}\n"
                context_text += f"Relevance: {doc['similarity_score']:.3f}\n\n"
            
            # Build the prompt
            system_prompt = """You are a helpful Teaching Assistant for the Tools in Data Science (TDS) course at IIT Madras. 
Your role is to provide accurate, helpful answers to student questions based on the course content and previous discussions.

Guidelines:
1. Use the provided context to answer questions accurately
2. If the question is about specific tools, models, or assignments, refer to the exact requirements mentioned in the context
3. Be helpful but concise
4. If you're not certain about something, acknowledge the uncertainty
5. For technical questions, provide practical guidance
6. Reference specific course materials when relevant"""

            user_prompt = f"""Question: {question}

{f"Image Description: {image_description}" if image_description else ""}

Context from course materials and discussions:
{context_text}

Please provide a helpful answer based on the context above. Be specific and practical in your response."""

            # Generate answer using OpenAI
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo-0125",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            answer = response.choices[0].message.content
            
            # Extract relevant discourse links
            links = self.extract_discourse_links(context_docs)
            
            return {
                "answer": answer,
                "links": links
            }
        
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "I'm sorry, I encountered an error while processing your question. Please try again or contact the course staff for assistance.",
                "links": []
            }

# Global instance for the API
answer_generator = None

def get_answer_generator():
    """Get the global answer generator instance"""
    global answer_generator
    if answer_generator is None:
        answer_generator = AnswerGenerator()
    return answer_generator

if __name__ == "__main__":
    # Test the answer generator
    generator = AnswerGenerator()
    
    test_question = "Should I use gpt-4o-mini which AI proxy supports, or gpt3.5 turbo?"
    result = generator.generate_answer(test_question)
    
    print("Question:", test_question)
    print("\nAnswer:", result["answer"])
    print("\nLinks:")
    for link in result["links"]:
        print(f"- {link['text']}: {link['url']}")
