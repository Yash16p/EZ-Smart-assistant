# 🚀 Smart Research Assistant v2.0

> **An intelligent research assistant powered by TinyLlama with Agentic AI capabilities, RAG, and fine-tuning support**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0+-green.svg)](https://python.langchain.com/)
[![TinyLlama](https://img.shields.io/badge/TinyLlama-1.1B-orange.svg)](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)

## ✨ Features

### 🤖 **AI-Powered Document Analysis**
- **TinyLlama Integration**: Ultra-lightweight 1.1B parameter model optimized for CPU
- **Smart Summarization**: Generate comprehensive document summaries in 150 words
- **Intelligent Q&A**: Ask questions and get context-aware answers
- **Challenge Mode**: Generate and evaluate comprehension questions

### 🔍 **Advanced RAG (Retrieval-Augmented Generation)**
- **Vector Search**: FAISS-based semantic document search
- **Context Analysis**: Intelligent relevance scoring and context evaluation
- **Source Tracking**: Page-level source snippets with metadata
- **Multi-format Support**: PDF and TXT document processing

### 🧠 **Agentic AI Capabilities**
- **Intelligent Tools**: Document search, context analysis, and answer generation
- **Conversation Memory**: Persistent chat history and context awareness
- **Workflow Orchestration**: LangGraph-powered decision-making workflows
- **Adaptive Responses**: Context-aware fallback and reasoning

### 🎯 **Fine-tuning & Optimization**
- **LoRA Integration**: Parameter-efficient fine-tuning on your documents
- **Training Data Collection**: Automatic Q&A pair collection during usage
- **CPU Optimization**: Designed for laptops without GPU requirements
- **Model Persistence**: Save and load fine-tuned models

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI        │    │   TinyLlama     │
│   Frontend      │◄──►│   Backend        │◄──►│   + RAG         │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   User          │    │   Agentic AI     │    │   Fine-tuning   │
│   Interface     │    │   Workflows      │    │   Manager       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- At least 2GB RAM (4GB recommended)
- Internet connection for first-time model download

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/smart-research-assistant.git
   cd smart-research-assistant
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start the application**
   ```bash
   # Terminal 1: Start FastAPI backend
   python main.py
   
   # Terminal 2: Start Streamlit frontend
   streamlit run streamlit_app.py
   ```

5. **Open your browser**
   - Frontend: http://localhost:8501
   - Backend API: http://localhost:8000

## 📖 Usage Guide

### 1. **Document Upload**
- Supported formats: PDF, TXT
- Automatic text extraction and cleaning
- Vector store creation for semantic search

### 2. **Document Analysis**
- **Summary**: Get a 150-word comprehensive summary
- **Q&A**: Ask questions about the document content
- **Challenge Mode**: Test your understanding with generated questions

### 3. **Agentic AI Features**
- **Smart Search**: Context-aware document retrieval
- **Relevance Analysis**: Automatic context quality assessment
- **Intelligent Answers**: Comprehensive responses with source references

### 4. **Fine-tuning**
- Automatic training data collection during usage
- LoRA-based fine-tuning on your documents
- Model persistence and loading

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the project root:

```env
# Model Configuration
MODEL_NAME=TinyLlama/TinyLlama-1.1B-Chat-v1.0
MAX_TOKENS=512
TEMPERATURE=0.1

# Fine-tuning Configuration
FINE_TUNE_EPOCHS=3
LEARNING_RATE=2e-4
LORA_RANK=8

# Vector Store Configuration
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
```

### Model Parameters
- **TinyLlama**: 1.1B parameters, optimized for CPU inference
- **Embeddings**: all-MiniLM-L6-v2 (384 dimensions)
- **Vector Store**: FAISS with CPU optimization

## 🧪 Testing

Run the test suite:

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=src
```

## 📊 Performance

### System Requirements
- **Minimum**: 2GB RAM, CPU-only
- **Recommended**: 4GB RAM, CPU-only
- **Optimal**: 8GB RAM, CPU-only

### Performance Metrics
- **Model Loading**: ~30 seconds (first time)
- **Inference Speed**: ~2-5 seconds per response
- **Memory Usage**: ~1.5GB during operation
- **Document Processing**: ~10-50 seconds (depending on size)

## 🛠️ Development

### Project Structure
```
smart-research-assistant/
├── src/
│   ├── __init__.py
│   ├── models.py          # Data models and schemas
│   ├── prompts.py         # Prompt templates
│   └── workflow.py        # Core AI logic and workflows
├── main.py                # FastAPI backend server
├── streamlit_app.py       # Streamlit frontend
├── requirements.txt       # Python dependencies
├── pyproject.toml        # Project configuration
└── README.md             # This file
```

### Key Components

#### **SmartAssistant Class**
- Main orchestrator for document analysis
- Integrates RAG, Agentic AI, and fine-tuning
- Manages conversation memory and context

#### **FineTuningManager Class**
- LoRA-based fine-tuning implementation
- Training data preparation and management
- Model persistence and loading

#### **Agentic AI Tools**
- `DocumentSearchTool`: Semantic document search
- `ContextAnalysisTool`: Relevance assessment
- `AnswerGenerationTool`: Intelligent answer generation

## 🔮 Roadmap

### v2.1 (Q1 2024)
- [ ] Multi-document support
- [ ] Advanced fine-tuning options
- [ ] Export/import fine-tuned models

### v2.2 (Q2 2024)
- [ ] Web scraping integration
- [ ] Citation generation
- [ ] Collaborative features

### v2.3 (Q3 2024)
- [ ] Multi-language support
- [ ] Advanced analytics dashboard
- [ ] API rate limiting and authentication

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **TinyLlama Team** for the ultra-lightweight model
- **LangChain Community** for the excellent framework
- **Hugging Face** for the transformers library
- **FAISS Team** for efficient vector search

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/smart-research-assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/smart-research-assistant/discussions)
- **Email**: contact@smartassistant.com

## ⚠️ Disclaimer

This is a research and educational tool. Please ensure you have the right to process any documents you upload. The AI-generated content should be reviewed for accuracy and appropriateness.

---

**Made with ❤️ for researchers, students, and knowledge seekers**
