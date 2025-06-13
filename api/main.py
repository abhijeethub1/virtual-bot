#!/usr/bin/env python3
"""
FastAPI main application for TDS Virtual TA
This is an alternative entry point that could be used instead of server.py
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Add the project root to the path
sys.path.append('/app')

from model.answer_generator import get_answer_generator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TDS Virtual Teaching Assistant",
    description="AI-powered teaching assistant for Tools in Data Science course",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QuestionRequest(BaseModel):
    question: str = Field(..., description="The student's question")
    image: Optional[str] = Field(None, description="Base64 encoded image (optional)")

class LinkResponse(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkResponse]

# Routes
@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "TDS Virtual TA API is running"}

@app.post("/", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Main endpoint for asking questions to the virtual TA
    
    This endpoint accepts a student question and optional base64 image,
    then returns an AI-generated answer with relevant links.
    """
    try:
        logger.info(f"Processing question: {request.question[:100]}...")
        
        # Get the answer generator
        generator = get_answer_generator()
        
        # Generate answer
        result = generator.generate_answer(
            question=request.question,
            image_base64=request.image
        )
        
        # Convert to response format
        links = [LinkResponse(url=link['url'], text=link['text']) 
                for link in result.get('links', [])]
        
        response = AnswerResponse(
            answer=result.get('answer', ''),
            links=links
        )
        
        logger.info("Question processed successfully")
        return response
    
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing your question"
        )

@app.get("/health")
async def health_check():
    """Detailed health check with system status"""
    try:
        generator = get_answer_generator()
        
        # Check if index is loaded
        index_loaded = generator.index is not None
        documents_loaded = len(generator.documents) > 0
        
        return {
            "status": "healthy",
            "index_loaded": index_loaded,
            "documents_count": len(generator.documents),
            "openai_configured": bool(os.getenv('OPENAI_API_KEY'))
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
