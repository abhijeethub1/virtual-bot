from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional
import uuid
from datetime import datetime
import sys

# Add project root to path for imports
sys.path.append('/app')

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI(
    title="TDS Virtual Teaching Assistant API",
    description="AI-powered teaching assistant for Tools in Data Science course",
    version="1.0.0"
)

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# Import TDS TA components (lazy import to avoid startup issues)
def get_answer_generator():
    try:
        from model.answer_generator import get_answer_generator
        return get_answer_generator()
    except Exception as e:
        logging.error(f"Error importing answer generator: {e}")
        return None

# Define Models
class StatusCheck(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StatusCheckCreate(BaseModel):
    client_name: str

class QuestionRequest(BaseModel):
    question: str = Field(..., description="The student's question")
    image: Optional[str] = Field(None, description="Base64 encoded image (optional)")

class LinkResponse(BaseModel):
    url: str
    text: str

class AnswerResponse(BaseModel):
    answer: str
    links: List[LinkResponse]

# Original routes
@api_router.get("/")
async def root():
    return {"message": "TDS Virtual Teaching Assistant API - Ready to help students!"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.dict()
    status_obj = StatusCheck(**status_dict)
    _ = await db.status_checks.insert_one(status_obj.dict())
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    status_checks = await db.status_checks.find().to_list(1000)
    return [StatusCheck(**status_check) for status_check in status_checks]

# New TDS Virtual TA routes
@api_router.post("/ask", response_model=AnswerResponse)
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
        if generator is None:
            raise HTTPException(
                status_code=503,
                detail="Virtual TA service is not available. Please ensure the index is built."
            )
        
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
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error while processing your question"
        )

@api_router.get("/health")
async def health_check():
    """Detailed health check with system status"""
    try:
        generator = get_answer_generator()
        
        if generator is None:
            return {
                "status": "unhealthy",
                "error": "Answer generator not available",
                "suggestion": "Run 'python /app/model/build_index.py' to build the index"
            }
        
        # Check if index is loaded
        index_loaded = generator.index is not None
        documents_loaded = len(generator.documents) > 0
        
        return {
            "status": "healthy" if index_loaded and documents_loaded else "partial",
            "index_loaded": index_loaded,
            "documents_count": len(generator.documents),
            "openai_configured": bool(os.getenv('OPENAI_API_KEY'))
        }
    
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
