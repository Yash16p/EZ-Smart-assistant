import uuid
import logging
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.models import (
    QuestionRequest, 
    ChallengeAnswer, 
    UploadResponse, 
    QuestionResponse, 
    EvaluationResponse,
    ConversationHistoryResponse,
    SourceSnippet
)
from src.workflow import SmartAssistant, safe_temp_file

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Smart Research Assistant API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://localhost:3000"], # Add other origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global storage for sessions
sessions: Dict[str, Dict[str, Any]] = {}

# Initialize assistant
assistant = SmartAssistant()


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """Upload and process document"""
    if file.content_type not in ["application/pdf", "text/plain"]:
        raise HTTPException(status_code=400, detail="Only PDF and TXT files are supported")
    
    session_id = str(uuid.uuid4())
    
    try:
        with safe_temp_file(suffix=f".{file.filename.split('.')[-1]}") as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file.flush()
            
            if file.content_type == "application/pdf":
                document_text = assistant.processor.extract_text_from_pdf(tmp_file.name)
            else:
                document_text = assistant.processor.extract_text_from_txt(tmp_file.name)
        
        if not document_text.strip():
            raise HTTPException(status_code=400, detail="No readable text found in the document")
        
        vector_store = assistant.processor.create_vector_store(document_text)
        summary = assistant.summarize_document(document_text)
        challenge_questions = assistant.generate_challenge_questions(document_text)
        
        sessions[session_id] = {
            "document_content": document_text,
            "vector_store": vector_store,
            "conversation_history": [],
            "document_summary": summary,
            "challenge_questions": challenge_questions,
            "workflow": assistant.create_workflow()
        }
        
        logger.info(f"Document uploaded successfully for session {session_id}")
        
        return UploadResponse(
            session_id=session_id,
            summary=summary,
            challenge_questions=challenge_questions,
            status="success"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """Ask a question about the document"""
    logger.info(f"Received question request for session {request.session_id}")
    
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions[request.session_id]
        
        if "vector_store" not in session:
            raise HTTPException(status_code=400, detail="Document not properly processed")
        
        # Initial state for the workflow
        state = {
            "current_question": request.question,
            "vector_store": session["vector_store"],
            "conversation_history": session.get("conversation_history", []),
            "document_content": session.get("document_content", ""),
            "messages": [],
        }
        
        # Invoke the workflow
        result = session["workflow"].invoke(state)
        
        # Update session history
        sessions[request.session_id]["conversation_history"] = result["conversation_history"]
        
        # Extract the last entry from history for the response
        last_entry = result["conversation_history"][-1]
        answer = last_entry.get("answer", "No answer found.")
        snippets_data = last_entry.get("source_snippets", [])
        
        # Convert snippet data to SourceSnippet models
        source_snippets = [SourceSnippet(**snippet) for snippet in snippets_data]

        logger.info(f"Question processed successfully. Answer: {answer[:50]}...")
        print(source_snippets)
        return QuestionResponse(
            answer=answer,
            session_id=request.session_id,
            source_snippets=source_snippets
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


@app.post("/challenge/answer", response_model=EvaluationResponse)
async def submit_challenge_answer(request: ChallengeAnswer):
    """Submit answer to challenge question"""
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        session = sessions[request.session_id]
        question_data = next((q for q in session["challenge_questions"] if q["id"] == request.question_id), None)
        
        if not question_data:
            raise HTTPException(status_code=404, detail="Question not found")
        
        evaluation = assistant.evaluate_answer(
            question_data["question"],
            question_data["expected_answer"],
            request.answer,
            session["document_content"]
        )
        
        return EvaluationResponse(evaluation=evaluation, question=question_data["question"])
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error evaluating answer: {e}")
        raise HTTPException(status_code=500, detail=f"Error evaluating answer: {str(e)}")


@app.get("/session/{session_id}/history", response_model=ConversationHistoryResponse)
async def get_conversation_history(session_id: str):
    """Get conversation history"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return ConversationHistoryResponse(
        history=sessions[session_id]["conversation_history"],
        summary=sessions[session_id]["document_summary"]
    )


@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Smart Research Assistant API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Smart Research Assistant API"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
