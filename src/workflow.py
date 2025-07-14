import os
import tempfile
import PyPDF2
import json
import contextlib
import time
import re
import unicodedata
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langgraph.graph import StateGraph, END
from fastapi import HTTPException

from .models import SessionState, SourceSnippet
from .prompts import Prompts
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Google Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    logger.warning("GOOGLE_API_KEY not found in environment variables.")

# Initialize Gemini models
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", # Using a newer model
    temperature=0.1,
    max_tokens=2048,
    timeout=30,
    max_retries=3,
)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


@contextlib.contextmanager
def safe_temp_file(suffix=""):
    """Context manager for safe temporary file handling on Windows"""
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        yield tmp_file
    finally:
        tmp_file.close()
        for attempt in range(3):
            try:
                os.unlink(tmp_file.name)
                break
            except (OSError, PermissionError):
                if attempt < 2:
                    time.sleep(0.1 * (attempt + 1))
                else:
                    print(f"Warning: Could not delete temporary file {tmp_file.name}")


def clean_text(text: str) -> str:
    """Clean text by removing problematic Unicode characters"""
    if not text:
        return ""
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    text = unicodedata.normalize('NFKD', text)
    text = re.sub(r'[^\x00-\x7F\u0080-\uFFFF]', '', text)
    text = re.sub(r'[\ud800-\udfff]', '', text)
    text = re.sub(r'[\ufffe\uffff]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class DocumentProcessor:
    """Handles document processing and vector store creation"""

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )

    def extract_text_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file with Unicode handling and page numbers"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for i, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text:
                        text += f"\n--- Page {i+1} ---\n" + clean_text(page_text) + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")
        return clean_text(text)

    def extract_text_from_txt(self, file_path: str) -> str:
        """Extract text from TXT file with Unicode handling"""
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as file:
                        return clean_text(file.read())
                except UnicodeDecodeError:
                    continue
            with open(file_path, 'rb') as file:
                return clean_text(file.read().decode('utf-8', errors='ignore'))
        except Exception as e:
            logger.error(f"Error extracting text file: {e}")
            raise HTTPException(status_code=400, detail=f"Error processing text file: {str(e)}")

    def create_vector_store(self, text: str) -> FAISS:
        """Create vector store from text with Unicode handling and metadata"""
        try:
            cleaned_text = clean_text(text)
            if not cleaned_text.strip():
                raise HTTPException(status_code=400, detail="No readable text found in document")

            # Split text into chunks while preserving page number context
            raw_documents = self.text_splitter.split_text(cleaned_text)
            
            processed_docs = []
            current_page = "N/A"
            for doc_content in raw_documents:
                page_match = re.search(r'--- Page (\d+) ---', doc_content)
                if page_match:
                    current_page = page_match.group(1)
                
                # Clean the content for the vector store, removing the page marker
                clean_content = re.sub(r'--- Page \d+ ---', '', doc_content).strip()

                if clean_content:
                    processed_docs.append(Document(
                        page_content=clean_content,
                        metadata={"page_number": current_page}
                    ))

            if not processed_docs:
                raise HTTPException(status_code=400, detail="No valid text chunks found after cleaning")

            return FAISS.from_documents(processed_docs, embeddings)
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


class SmartAssistant:
    """Main assistant class that handles document analysis and question answering"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.prompts = Prompts()

    def create_workflow(self) -> StateGraph:
        """Create LangGraph workflow with more reasoning steps"""
        workflow = StateGraph(SessionState)

        # 1. Retrieve Context
        def retrieve_context(state: SessionState):
            logger.info("Workflow Step: Retrieving Context")
            question = state["current_question"]
            vector_store = state["vector_store"]
            docs = vector_store.similarity_search(question, k=5)
            return {"retrieved_docs": docs}

        # 2. Grade Context Relevance
        def grade_context(state: SessionState):
            logger.info("Workflow Step: Grading Context Relevance")
            question = state["current_question"]
            docs = state["retrieved_docs"]
            context = "\n\n".join([doc.page_content for doc in docs])
            
            prompt = self.prompts.CONTEXT_GRADING_PROMPT.format(context=context, question=question)
            response = llm.invoke(prompt)
            grade = response.content.strip().lower()
            
            logger.info(f"Context relevance grade: {grade}")
            return {"context_is_relevant": grade == "yes"}

        # 3. Generate Answer
        def generate_answer(state: SessionState):
            logger.info("Workflow Step: Generating Answer")
            question = state["current_question"]
            docs = state["retrieved_docs"]
            history = state["conversation_history"]
            
            context = "\n\n".join([f"Page {doc.metadata.get('page_number', 'N/A')}: {doc.page_content}" for doc in docs])
            history_str = "\n".join([f"Q: {h['question']}\nA: {h['answer']}" for h in history[-3:]])

            prompt = self.prompts.QA_PROMPT.format(context=context, question=question, history=history_str)
            response = llm.invoke(prompt)
            
            try:
                # Clean response to extract JSON
                content = response.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3]
                
                result_json = json.loads(content)
                answer = result_json.get("answer", "Could not generate an answer.")
                
                # Create SourceSnippet objects
                source_snippets = []
                for doc in docs:
                    for snippet_text in result_json.get("source_snippets", []):
                        if snippet_text in doc.page_content:
                             source_snippets.append(SourceSnippet(
                                 page_number=doc.metadata.get('page_number', 'N/A'),
                                 content=snippet_text
                             ))
                             break # Move to next doc once a match is found for a snippet

                # Deduplicate snippets
                unique_snippets = {s.content: s for s in source_snippets}.values()

            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Error parsing LLM response for answer: {e}")
                answer = "I apologize, but I encountered an error generating a structured response. Please try again."
                unique_snippets = []

            updated_history = history.copy()
            updated_history.append({
                "question": question,
                "answer": answer,
                "source_snippets": [s.dict() for s in unique_snippets],
                "timestamp": datetime.now().isoformat()
            })

            return {
                "current_answer": answer, 
                "conversation_history": updated_history,
                "retrieved_docs": docs # Pass along for the final response
            }
            
        # 4. Fallback for irrelevant context
        def fallback_answer(state: SessionState):
            logger.info("Workflow Step: Fallback Answer (Irrelevant Context)")
            return {"current_answer": "I'm sorry, but I couldn't find a relevant answer in the document for your question."}

        # 5. Define Conditional Edges
        def decide_to_generate(state: SessionState):
            logger.info("Workflow Step: Deciding to Generate Answer")
            if state["context_is_relevant"]:
                logger.info("Decision: Context is relevant. Generating answer.")
                return "generate_answer"
            else:
                logger.info("Decision: Context is not relevant. Using fallback.")
                return "fallback_answer"

        workflow.add_node("retrieve_context", retrieve_context)
        workflow.add_node("grade_context", grade_context)
        workflow.add_node("generate_answer", generate_answer)
        workflow.add_node("fallback_answer", fallback_answer)

        workflow.set_entry_point("retrieve_context")
        workflow.add_edge("retrieve_context", "grade_context")
        workflow.add_conditional_edges(
            "grade_context",
            decide_to_generate,
            {
                "generate_answer": "generate_answer",
                "fallback_answer": "fallback_answer",
            },
        )
        workflow.add_edge("generate_answer", END)
        workflow.add_edge("fallback_answer", END)

        return workflow.compile()

    def summarize_document(self, document_text: str) -> str:
        """Generate document summary"""
        try:
            cleaned_text = clean_text(document_text)
            prompt = self.prompts.SUMMARIZER_PROMPT.format(document=cleaned_text[:8000])
            response = llm.invoke(prompt)
            return clean_text(response.content)
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return "Error generating summary."

    def generate_challenge_questions(self, document_text: str) -> List[Dict[str, Any]]:
        """Generate challenge questions"""
        try:
            cleaned_text = clean_text(document_text)
            prompt = self.prompts.CHALLENGE_PROMPT.format(document=cleaned_text[:8000])
            response = llm.invoke(prompt)
            content = clean_text(response.content).strip()
            if content.startswith("```json"):
                content = content[7:-3]
            questions = json.loads(content)
            for q in questions:
                if "id" not in q:
                    q["id"] = str(uuid.uuid4())
            return questions
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error generating challenge questions: {e}")
            return self.prompts.FALLBACK_QUESTIONS

    def evaluate_answer(self, question: str, expected_answer: str, user_answer: str, document_text: str) -> Dict[str, Any]:
        """Evaluate user's answer"""
        try:
            context = clean_text(document_text[:4000]) # Provide context for evaluation
            prompt = self.prompts.EVALUATION_PROMPT.format(
                context=context,
                question=clean_text(question),
                expected_answer=clean_text(expected_answer),
                user_answer=clean_text(user_answer)
            )
            response = llm.invoke(prompt)
            content = clean_text(response.content).strip()
            if content.startswith("```json"):
                content = content[7:-3]
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"Error evaluating answer: {e}")
            return {
                "score": 50,
                "feedback": "Unable to evaluate answer automatically.",
                "document_reference": "N/A",
                "suggestions": "Please refer to the document."
            }
