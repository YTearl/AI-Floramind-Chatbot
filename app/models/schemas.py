from typing import List, Optional
from pydantic import BaseModel, Field

class ChatMessage(BaseModel):
    """Single turn message in the conversation history."""
    role: str = Field(..., description="Role of the sender: 'user' or 'assistant'/'model'")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    """Payload sent by the frontend chat interface."""
    message: str = Field(..., min_length=1, max_length=2000, description="User's gardening question")
    history: Optional[List[ChatMessage]] = Field(
        default_factory=list, description="Prior conversation turns"
    )
    top_k: Optional[int] = Field(default=5, ge=1, le=10, description="Number of references to retrieve")

class CitationSource(BaseModel):
    """Structured citation metadata returned alongside the generated answer."""
    id: str = Field(..., description="Document ID")
    title: str = Field(..., description="Title of the gardening document")
    category: Optional[str] = Field(None, description="Category of the gardening knowledge")
    section: Optional[str] = Field(None, description="Section or subsection")
    snippet: str = Field(..., description="Excerpt from the source document")
    score: float = Field(..., description="Similarity score from vector search (0.0 to 1.0)")

class ChatResponse(BaseModel):
    """Response returned to the frontend."""
    answer: str = Field(..., description="Generated answer from Gemini")
    citations: List[CitationSource] = Field(
        default_factory=list, description="Verified source documents supporting the answer"
    )
    model_used: str = Field(..., description="Identifier of the Gemini model used")
    retrieval_status: str = Field(
        ...,
        description="Status of knowledge retrieval: 'grounded', 'insufficient', 'offline_fallback', 'mock'",
    )
    is_grounded: bool = Field(
        ..., description="True if answer is backed by relevant knowledge base references"
    )

class HealthResponse(BaseModel):
    """System health check and configuration status."""
    status: str
    gemini_configured: bool
    gemini_model: str
    pinecone_configured: bool
    pinecone_index: str
    embedding_model: str

class IndexStatusResponse(BaseModel):
    """Response for vector database indexing action."""
    status: str
    indexed_count: int
    message: str
