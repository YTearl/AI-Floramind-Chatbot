import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import settings
from app.models.schemas import ChatRequest, ChatResponse, HealthResponse, IndexStatusResponse
from app.services.rag_service import get_rag_service
from scripts.index_knowledge import index_gardening_knowledge

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("floramind")

# Initialize FastAPI application
app = FastAPI(
    title="Floramind Gardening Assistant",
    description="RAG gardening chatbot powered by Google Gemini, Hugging Face, and Pinecone.",
    version="2.0.0",
)

# CORS setup for flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static and Templates configuration
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


@app.get("/", response_class=HTMLResponse)
async def serve_home(request: Request):
    """Renders the Floramind Chatbot interface."""
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "gemini_model": settings.GEMINI_MODEL,
            "embedding_model": settings.HF_EMBEDDING_MODEL,
            "pinecone_configured": settings.is_pinecone_configured,
            "gemini_configured": settings.is_gemini_configured,
        },
    )


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Processes user gardening queries through the RAG pipeline."""
    try:
        rag_svc = get_rag_service()
        response = rag_svc.answer_question(
            query=request.message,
            history=request.history,
            top_k=request.top_k or settings.TOP_K_RESULTS,
        )
        return response
    except Exception as e:
        logger.error(f"Error handling chat query: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating your answer: {str(e)}",
        )


@app.get("/api/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Returns application health and configuration flags without revealing secrets."""
    return HealthResponse(
        status="healthy",
        gemini_configured=settings.is_gemini_configured,
        gemini_model=settings.GEMINI_MODEL,
        pinecone_configured=settings.is_pinecone_configured,
        pinecone_index=settings.PINECONE_INDEX_NAME,
        embedding_model=settings.HF_EMBEDDING_MODEL,
    )


@app.post("/api/index-knowledge", response_model=IndexStatusResponse)
async def trigger_indexing() -> IndexStatusResponse:
    """Indexes the local gardening dataset into the configured vector database."""
    try:
        count = index_gardening_knowledge()
        return IndexStatusResponse(
            status="success",
            indexed_count=count,
            message=f"Successfully indexed {count} gardening documents.",
        )
    except Exception as e:
        logger.error(f"Error during manual indexing: {e}")
        return IndexStatusResponse(
            status="error",
            indexed_count=0,
            message=f"Indexing failed: {str(e)}",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
