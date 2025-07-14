# 🔍 Smart Research Assistant

[![Demo Video](https://img.shields.io/badge/Watch-Demo-red?logo=youtube)](https://youtu.be/3qAPiV4LkL0)

An AI-powered document analysis tool that helps you interact with research papers, reports, and documents through intelligent questioning and comprehension testing.

## ✨ Features

- **📄 Document Processing**: Upload PDF and TXT files for analysis
- **🤔 Intelligent Q&A**: Ask questions about your documents with contextual answers
- **🧠 Challenge Mode**: Test your understanding with AI-generated questions
- **💭 Conversation Memory**: Maintains context across multiple questions
- **📊 Answer Evaluation**: Detailed feedback on challenge question responses
- **🎯 Source References**: Answers include relevant document snippets with page numbers
- **📱 Modern UI**: Responsive web interface with dark mode support

## 🏗️ Architecture

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit UI  │───▶│   FastAPI       │───▶│   LangChain     │
│   (Frontend)    │    │   (Backend)     │    │   (Processing)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Session  │    │   Document      │    │   Vector Store  │
│   Management    │    │   Processing    │    │   (FAISS)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Technology Stack

**Backend Components:**
- **FastAPI**: RESTful API server
- **LangChain**: Document processing and LLM integration
- **LangGraph**: Workflow management and reasoning chains
- **Google Gemini**: AI model for text generation and embeddings
- **FAISS**: Vector database for semantic search
- **PyPDF2**: PDF text extraction

**Frontend Components:**
- **Streamlit**: Web application framework
- **Custom CSS**: Modern UI styling with responsive design
- **Session State Management**: Conversation history and user context

## 📁 Project Structure

```
smart-research-assistant/
├── src/
│   ├── __init__.py
│   ├── models.py          # Pydantic models for API requests/responses
│   ├── workflow.py        # Core AI workflow and document processing
│   └── prompts.py         # LLM prompts and templates
├── main.py                # FastAPI backend server
├── streamlit_app.py       # Streamlit frontend application
├── requirements.txt       # Python dependencies
├── .env.example          # Environment variables template
├── .env                  # Environment variables (create this)
├── pyproject.toml        # UV package configuration
└── README.md             # This file
```

### Key Components

#### 1. `main.py` - FastAPI Backend
- **Session Management**: Handles user sessions and document storage
- **API Endpoints**: RESTful endpoints for document upload, Q&A, and challenges
- **CORS Configuration**: Enables frontend-backend communication
- **Error Handling**: Comprehensive error handling and logging

#### 2. `streamlit_app.py` - Frontend Interface
- **Multi-mode Interface**: Upload, Summary, Q&A, and Challenge modes
- **Responsive Design**: Modern CSS with dark mode support
- **Session State**: Maintains conversation history and user context
- **Real-time Updates**: Dynamic UI updates based on user interactions

#### 3. `src/workflow.py` - AI Processing Engine
- **Document Processing**: PDF/TXT text extraction and cleaning
- **Vector Store Creation**: FAISS-based semantic search setup
- **LangGraph Workflow**: Multi-step reasoning chain for Q&A
- **Smart Assistant**: Main orchestrator for all AI operations

#### 4. `src/models.py` - Data Models
- **Request/Response Models**: Pydantic models for API validation
- **Session State**: TypedDict for workflow state management
- **Data Structures**: Models for questions, answers, and evaluations

## 🚀 Setup Instructions

### Prerequisites
- Python 3.8+
- UV package manager (recommended) or pip
- Google Gemini API key

### 1. Clone the Repository
```bash
git clone <repository-url>
cd smart-research-assistant
```

### 2. Initialize Project with UV
```bash
# Install UV if not already installed
pip install uv

# Initialize the project
uv init

# Install dependencies
uv add -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory:
```bash
cp .env.example .env
```

Edit the `.env` file and add your Google Gemini API key:
```env
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 4. Get Google Gemini API Key
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the key to your `.env` file

### 5. Run the Application

#### Backend Server
```bash
uv run fastapi dev main.py
```
The API will be available at `http://localhost:8000`

#### Frontend Application
```bash
uv run streamlit run streamlit_app.py
```
The web interface will be available at `http://localhost:8501`

### Alternative Setup (without UV)
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run applications
python -m uvicorn main:app --reload --port 8000
streamlit run streamlit_app.py
```

## 🔄 How It Works

### Document Processing Flow
1. **Upload**: User uploads PDF/TXT document
2. **Text Extraction**: PyPDF2 extracts text with page numbers
3. **Text Cleaning**: Unicode normalization and cleaning
4. **Chunking**: RecursiveCharacterTextSplitter creates manageable chunks
5. **Vectorization**: Google Gemini embeddings create vector representations
6. **Storage**: FAISS vector store enables semantic search

### Question Answering Workflow (LangGraph)
```
┌─────────────────┐
│ User Question   │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Retrieve Context│ ← Vector similarity search
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Grade Relevance │ ← LLM evaluates context quality
└─────────┬───────┘
          │
      ┌───▼───┐
      │ Relevant? │
      └───┬───┘
    Yes   │   No
          │
    ┌─────▼──────┐    ┌─────────────┐
    │ Generate   │    │ Fallback    │
    │ Answer     │    │ Response    │
    └─────┬──────┘    └─────────────┘
          │
          ▼
┌─────────────────┐
│ Structured      │
│ Response        │
└─────────────────┘
```

### Challenge Mode Process
1. **Question Generation**: AI analyzes document and creates questions
2. **Difficulty Levels**: Easy, Medium, Hard questions generated
3. **Answer Evaluation**: Multi-criteria assessment of user responses
4. **Feedback Loop**: Detailed feedback with improvement suggestions

## 📊 API Endpoints

### Document Upload
```
POST /upload
Content-Type: multipart/form-data
```

### Ask Question
```
POST /ask
Content-Type: application/json
{
  "question": "What is the main argument?",
  "session_id": "session_uuid"
}
```

### Submit Challenge Answer
```
POST /challenge/answer
Content-Type: application/json
{
  "question_id": "question_uuid",
  "answer": "User's answer",
  "session_id": "session_uuid"
}
```

### Get Conversation History
```
GET /session/{session_id}/history
```

## 🎯 Usage Examples

### 1. Basic Q&A
```python
# Upload document
response = requests.post("/upload", files={"file": document})
session_id = response.json()["session_id"]

# Ask question
response = requests.post("/ask", json={
    "question": "What are the main findings?",
    "session_id": session_id
})
```

### 2. Challenge Mode
```python
# Get challenge questions from upload response
questions = upload_response["challenge_questions"]

# Submit answer
response = requests.post("/challenge/answer", json={
    "question_id": questions[0]["id"],
    "answer": "My detailed answer",
    "session_id": session_id
})
```

## 🔧 Configuration

### Environment Variables
- `GOOGLE_API_KEY`: Required for Gemini API access
- `LANGCHAIN_TRACING_V2`: Optional for LangChain debugging
- `LANGCHAIN_API_KEY`: Optional for LangSmith tracing

### Customization Options
- **Chunk Size**: Modify `chunk_size` in `DocumentProcessor`
- **Model Temperature**: Adjust `temperature` in `ChatGoogleGenerativeAI`
- **Vector Search Results**: Change `k` parameter in similarity search
- **UI Theme**: Modify CSS variables in `streamlit_app.py`

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: GOOGLE_API_KEY not found
   Solution: Ensure .env file exists with correct API key
   ```

2. **Document Processing Fails**
   ```
   Error: No readable text found
   Solution: Check document format and content
   ```

3. **Port Already in Use**
   ```
   Error: Port 8000 already in use
   Solution: Use different port or kill existing process
   ```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **LangChain**: For powerful document processing capabilities
- **Google Gemini**: For advanced AI reasoning and embeddings
- **Streamlit**: For rapid web application development
- **FastAPI**: For robust API development

---

For more information or support, please open an issue on the GitHub repository.
