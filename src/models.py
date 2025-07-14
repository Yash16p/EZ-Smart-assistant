from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langchain.schema import Document


class QuestionRequest(BaseModel):
    """Request model for asking questions about a document"""
    question: str
    session_id: str


class ChallengeAnswer(BaseModel):
    """Model for submitting answers to challenge questions"""
    question_id: str
    answer: str
    session_id: str


class UploadResponse(BaseModel):
    """Response model for document upload"""
    session_id: str
    summary: str
    challenge_questions: List[Dict[str, Any]]
    status: str


class SourceSnippet(BaseModel):
    """Model for a single source snippet"""
    page_number: str
    content: str

class QuestionResponse(BaseModel):
    """Response model for question answering"""
    answer: str
    session_id: str
    source_snippets: List[SourceSnippet]


class EvaluationResponse(BaseModel):
    """Response model for challenge answer evaluation"""
    evaluation: Dict[str, Any]
    question: str


class ConversationHistoryResponse(BaseModel):
    """Response model for conversation history"""
    history: List[Dict[str, Any]]
    summary: str


class SessionState(TypedDict):
    """State type for LangGraph workflow"""
    messages: Annotated[list, add_messages]
    document_content: str
    vector_store: Optional[object]
    conversation_history: list
    document_summary: str
    challenge_questions: list
    current_question: str
    current_answer: str
    retrieved_docs: Optional[List[Document]] # To hold retrieved documents
    context_is_relevant: bool # To hold the result of the relevance check


class ConversationEntry(BaseModel):
    """Model for individual conversation entries"""
    question: str
    answer: str
    source_snippets: List[SourceSnippet]
    timestamp: str


class ChallengeQuestion(BaseModel):
    """Model for challenge questions"""
    id: str
    question: str
    expected_answer: str
    difficulty: str


class AnswerEvaluation(BaseModel):
    """Model for answer evaluation results"""
    score: int
    feedback: str
    document_reference: str
    suggestions: str


class SessionData(BaseModel):
    """Model for session data storage"""
    document_content: str
    vector_store: Optional[Any]
    conversation_history: List[ConversationEntry]
    document_summary: str
    challenge_questions: List[ChallengeQuestion]
    workflow: Optional[Any]
