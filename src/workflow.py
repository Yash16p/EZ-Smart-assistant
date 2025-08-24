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
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Custom exception for error handling
class HTTPException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)

# Hugging Face imports for TinyLlama
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, SystemMessage

# Fine-tuning imports
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset
import torch
from torch.utils.data import DataLoader

from .models import SessionState, SourceSnippet
from .prompts import Prompts
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Global variables for lazy initialization
llm = None
tokenizer = None
base_model = None
embeddings = None
finetuning_manager = None

# Initialize TinyLlama model
def get_tinyllama_model():
    """Initialize TinyLlama model optimized for CPU"""
    global llm, tokenizer, base_model
    
    if llm is not None:
        return llm, tokenizer, base_model
        
    try:
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with CPU optimizations
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for CPU
            device_map="cpu",
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # Create transformers pipeline first
        from transformers import pipeline
        transformers_pipeline = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            return_full_text=False
        )
        
        # Create LangChain pipeline
        pipeline = HuggingFacePipeline(
            pipeline=transformers_pipeline
        )
        
        llm = pipeline
        base_model = model
        
        logger.info("✅ TinyLlama model loaded successfully")
        return llm, tokenizer, base_model
        
    except Exception as e:
        logger.error(f"❌ Error loading TinyLlama: {e}")
        raise Exception(f"Failed to load TinyLlama model: {str(e)}")

def get_embeddings():
    """Initialize CPU-friendly embeddings"""
    global embeddings
    
    if embeddings is not None:
        return embeddings
        
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        logger.error(f"❌ Error loading embeddings: {e}")
        raise Exception(f"Failed to load embeddings: {str(e)}")

def get_finetuning_manager():
    """Initialize fine-tuning manager"""
    global finetuning_manager
    
    if finetuning_manager is not None:
        return finetuning_manager
        
    try:
        finetuning_manager = FineTuningManager()
        return finetuning_manager
    except Exception as e:
        logger.error(f"❌ Error creating fine-tuning manager: {e}")
        raise Exception(f"Failed to create fine-tuning manager: {str(e)}")

# Fine-tuning configuration
class FineTuningManager:
    """Manages fine-tuning of TinyLlama on document data"""
    
    def __init__(self):
        self.lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        )
        self.fine_tuned_model = None
    
    def prepare_training_data(self, documents: List[str], questions: List[str], answers: List[str]) -> Dataset:
        """Prepare training data for fine-tuning"""
        try:
            # Get tokenizer
            _, tokenizer, _ = get_tinyllama_model()
            
            # Format data for instruction fine-tuning
            training_data = []
            for doc, q, a in zip(documents, questions, answers):
                # Create instruction format
                instruction = f"""Context: {doc[:1000]}

Question: {q}

Answer: {a}"""
                
                training_data.append({
                    "text": instruction,
                    "input_ids": tokenizer.encode(instruction, truncation=True, max_length=512)
                })
            
            return Dataset.from_list(training_data)
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            raise
    
    def fine_tune_model(self, training_data: Dataset, epochs: int = 3) -> bool:
        """Fine-tune the model on prepared data"""
        try:
            logger.info("🚀 Starting model fine-tuning...")
            
            # Get base model
            _, _, base_model = get_tinyllama_model()
            
            # Apply LoRA to base model
            model = get_peft_model(base_model, self.lora_config)
            
            # Training arguments optimized for CPU
            training_args = TrainingArguments(
                output_dir="./fine_tuned_model",
                num_train_epochs=epochs,
                per_device_train_batch_size=1,  # Small batch size for CPU
                gradient_accumulation_steps=4,
                warmup_steps=100,
                learning_rate=2e-4,
                fp16=False,  # Disable fp16 for CPU
                logging_steps=10,
                save_steps=100,
                eval_steps=100,
                evaluation_strategy="steps",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                greater_is_better=False,
                dataloader_pin_memory=False,  # Disable for CPU
                remove_unused_columns=False
            )
            
            # Create trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=training_data,
                tokenizer=tokenizer
            )
            
            # Start training
            logger.info("📚 Training started...")
            trainer.train()
            
            # Save the fine-tuned model
            trainer.save_model()
            logger.info("✅ Model fine-tuning completed successfully!")
            
            # Load fine-tuned model for inference
            self.fine_tuned_model = model
            return True
            
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            return False
    
    def get_fine_tuned_llm(self):
        """Get the fine-tuned model for inference"""
        if self.fine_tuned_model:
            return HuggingFacePipeline(
                pipeline=self.fine_tuned_model,
                tokenizer=tokenizer,
                max_new_tokens=512,
                temperature=0.1
            )
        return get_tinyllama_model()[0]  # Fallback to base model

# Agentic AI Tools
class DocumentSearchTool(BaseTool):
    name: str = "document_search"
    description: str = "Search for relevant information in the uploaded document"
    
    def _run(self, query: str, vector_store) -> str:
        """Search document using vector similarity"""
        try:
            docs = vector_store.similarity_search(query, k=3)
            return "\n\n".join([f"Page {doc.metadata.get('page_number', 'N/A')}: {doc.page_content}" for doc in docs])
        except Exception as e:
            return f"Search error: {str(e)}"
    
    def _arun(self, query: str, vector_store):
        return self._run(query, vector_store)

class ContextAnalysisTool(BaseTool):
    name: str = "context_analysis"
    description: str = "Analyze the relevance and quality of retrieved context for a question"
    
    def _run(self, context: str, question: str) -> str:
        """Analyze context relevance"""
        try:
            # Get LLM
            llm, _, _ = get_tinyllama_model()
            
            prompt = f"""<|system|>
You are a context analysis expert. Analyze if the provided context is relevant to answer the question.

Question: {question}
Context: {context}

Provide a brief analysis (2-3 sentences) of:
1. Relevance to the question
2. Completeness of information
3. Any gaps or limitations

Analysis:</s>
<|user|>
Please analyze the context relevance.</s>
<|assistant|>"""
            
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Analysis error: {str(e)}"
    
    def _arun(self, context: str, question: str):
        return self._run(context, question)

class AnswerGenerationTool(BaseTool):
    name: str = "answer_generation"
    description: str = "Generate comprehensive answers based on document context and conversation history"
    
    def _run(self, context: str, question: str, history: str) -> str:
        """Generate structured answer"""
        try:
            # Get LLM
            llm, _, _ = get_tinyllama_model()
            
            prompt = f"""<|system|>
You are an expert document analyst. Generate a comprehensive answer to the question based on the provided context.

Context: {context}
Question: {question}
Previous Conversation: {history}

Requirements:
1. Base your answer ONLY on the provided context
2. Provide specific references to document sections
3. If information is missing, clearly state what cannot be answered
4. Structure your response logically

Answer:</s>
<|user|>
Please answer the question based on the context.</s>
<|assistant|>"""
            
            response = llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return f"Answer generation error: {str(e)}"
    
    def _arun(self, context: str, question: str, history: str):
        return self._run(context, question, history)

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

            # Get embeddings
            embeddings = get_embeddings()
            return FAISS.from_documents(processed_docs, embeddings)
        except Exception as e:
            logger.error(f"Error creating vector store: {e}")
            raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")

class SmartAssistant:
    """Main assistant class that handles document analysis and question answering with Agentic AI and Fine-tuning"""

    def __init__(self):
        self.processor = DocumentProcessor()
        self.prompts = Prompts()
        self.tools = [
            DocumentSearchTool(),
            ContextAnalysisTool(),
            AnswerGenerationTool()
        ]
        
        # Initialize agentic memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Training data storage for fine-tuning
        self.training_data = {
            "documents": [],
            "questions": [],
            "answers": []
        }

    def create_agent(self, vector_store) -> Any:
        """Create an agentic AI system for document analysis"""
        try:
            # Get LLM
            llm, _, _ = get_tinyllama_model()
            
            # Create agent with tools
            agent = initialize_agent(
                tools=self.tools,
                llm=llm,
                agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                memory=self.memory,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3
            )
            return agent
        except Exception as e:
            logger.error(f"Error creating agent: {e}")
            return None

    def create_workflow(self):
        """Create simplified workflow without LangGraph"""
        # Simple workflow implementation - no complex state management needed
        return None

    def fine_tune_on_documents(self, epochs: int = 3) -> bool:
        """Fine-tune the model on collected document data"""
        try:
            if len(self.training_data["documents"]) < 5:
                logger.warning("Not enough training data. Need at least 5 Q&A pairs.")
                return False
            
            logger.info(f"🎯 Starting fine-tuning with {len(self.training_data['documents'])} training examples")
            
            # Get fine-tuning manager
            finetuning_manager = get_finetuning_manager()
            
            # Prepare training data
            training_dataset = finetuning_manager.prepare_training_data(
                self.training_data["documents"],
                self.training_data["questions"],
                self.training_data["answers"]
            )
            
            # Start fine-tuning
            success = finetuning_manager.fine_tune_model(training_dataset, epochs)
            
            if success:
                # Update LLM to use fine-tuned model
                global llm
                llm = finetuning_manager.get_fine_tuned_llm()
                logger.info("✅ Fine-tuned model is now active!")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Fine-tuning failed: {e}")
            return False

    def _evaluate_relevance(self, analysis: str) -> float:
        """Evaluate context relevance from analysis"""
        try:
            # Simple keyword-based relevance scoring
            positive_keywords = ["relevant", "sufficient", "complete", "helpful", "adequate"]
            negative_keywords = ["irrelevant", "insufficient", "incomplete", "unhelpful", "limited"]
            
            analysis_lower = analysis.lower()
            positive_score = sum(1 for word in positive_keywords if word in analysis_lower)
            negative_score = sum(1 for word in negative_keywords if word in analysis_lower)
            
            if positive_score == 0 and negative_score == 0:
                return 0.5  # Neutral
            
            total_score = positive_score + negative_score
            relevance = positive_score / total_score if total_score > 0 else 0.5
            
            return min(max(relevance, 0.0), 1.0)
        except Exception as e:
            logger.error(f"Error evaluating relevance: {e}")
            return 0.5

    def _extract_source_snippets(self, context: str, answer: str) -> List[Dict[str, Any]]:
        """Extract source snippets from context"""
        try:
            # Simple snippet extraction - split context into sentences
            sentences = re.split(r'[.!?]+', context)
            relevant_snippets = []
            
            # Find sentences that contain key terms from the answer
            answer_words = set(re.findall(r'\b\w+\b', answer.lower()))
            
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 20:  # Only meaningful sentences
                    sentence_words = set(re.findall(r'\b\w+\b', sentence.lower()))
                    overlap = len(answer_words.intersection(sentence_words))
                    
                    if overlap > 2:  # At least 3 word overlap
                        # Extract page number if available
                        page_match = re.search(r'Page (\d+):', sentence)
                        page_number = page_match.group(1) if page_match else "N/A"
                        
                        relevant_snippets.append({
                            "page_number": page_number,
                            "content": sentence[:200] + "..." if len(sentence) > 200 else sentence
                        })
            
            return relevant_snippets[:3]  # Return top 3 snippets
        except Exception as e:
            logger.error(f"Error extracting source snippets: {e}")
            return []

    def summarize_document(self, document_text: str) -> str:
        """Generate document summary using TinyLlama"""
        try:
            cleaned_text = clean_text(document_text)
            
            # Get LLM
            llm, _, _ = get_tinyllama_model()
            
            # Create a specialized summarization prompt for TinyLlama
            summary_prompt = f"""<|system|>
You are an expert document analyst. Create a comprehensive summary of this document in exactly 150 words.

Requirements:
1. Identify the main thesis/argument
2. Highlight key findings and evidence
3. Note methodology and approach
4. Mention limitations or counterarguments
5. Use precise, academic language

Document: {cleaned_text[:8000]}

Summary:</s>
<|user|>
Please summarize this document.</s>
<|assistant|>"""
            
            response = llm.invoke(summary_prompt)
            return clean_text(response.content if hasattr(response, 'content') else str(response))
        except Exception as e:
            logger.error(f"Error summarizing document: {e}")
            return "Error generating summary."

    def generate_challenge_questions(self, document_text: str) -> List[Dict[str, Any]]:
        """Generate challenge questions using TinyLlama"""
        try:
            cleaned_text = clean_text(document_text)
            
            # Get LLM
            llm, _, _ = get_tinyllama_model()
            
            challenge_prompt = f"""<|system|>
Generate exactly 3 sophisticated questions that test understanding of this document.

Requirements:
1. Questions must be answerable ONLY from the document
2. Include different difficulty levels (easy, medium, hard)
3. Test analysis, synthesis, and evaluation skills
4. Focus on key concepts and arguments

Document: {cleaned_text[:8000]}

Return as JSON array:
[
    {{
        "id": "q1",
        "question": "Your question here",
        "expected_answer": "Expected answer with reasoning",
        "difficulty": "easy/medium/hard"
    }}
]

Questions:</s>
<|user|>
Please generate challenge questions.</s>
<|assistant|>"""
            
            response = llm.invoke(challenge_prompt)
            content = clean_text(response.content if hasattr(response, 'content') else str(response))
            
            # Clean and parse JSON response
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            try:
                questions = json.loads(content)
                for q in questions:
                    if "id" not in q:
                        q["id"] = str(uuid.uuid4())
                return questions
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON response, using fallback questions")
                return self.prompts.FALLBACK_QUESTIONS
                
        except Exception as e:
            logger.error(f"Error generating challenge questions: {e}")
            return self.prompts.FALLBACK_QUESTIONS

    def evaluate_answer(self, question: str, expected_answer: str, user_answer: str, document_text: str) -> Dict[str, Any]:
        """Evaluate user's answer using TinyLlama"""
        try:
            context = clean_text(document_text[:4000])
            
            # Get LLM
            llm, _, _ = get_tinyllama_model()
            
            evaluation_prompt = f"""<|system|>
Evaluate this user's answer to a challenge question.

Document Context: {context}
Question: {question}
Expected Answer: {expected_answer}
User Answer: {user_answer}

Provide evaluation in JSON format:
{{
    "score": <0-100>,
    "feedback": "Detailed feedback",
    "document_reference": "Specific document section",
    "suggestions": "Improvement suggestions"
}}

Evaluation:</s>
<|user|>
Please evaluate this answer.</s>
<|assistant|>"""
            
            response = llm.invoke(evaluation_prompt)
            content = clean_text(response.content if hasattr(response, 'content') else str(response))
            
            # Clean and parse JSON response
            if content.startswith("```json"):
                content = content[7:-3]
            elif content.startswith("```"):
                content = content[3:-3]
            
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                logger.warning("Failed to parse evaluation JSON, using fallback")
                return {
                    "score": 50,
                    "feedback": "Unable to evaluate answer automatically.",
                    "document_reference": "N/A",
                    "suggestions": "Please refer to the document."
                }
                
        except Exception as e:
            logger.error(f"Error evaluating answer: {e}")
            return {
                "score": 50,
                "feedback": "Unable to evaluate answer automatically.",
                "document_reference": "N/A",
                "suggestions": "Please refer to the document."
            }
