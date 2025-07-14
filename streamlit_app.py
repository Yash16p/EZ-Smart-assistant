import streamlit as st
import requests
import json
from typing import Dict, List
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Smart Research Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Improved custom CSS with modern color scheme
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Root variables for consistent theming */
    :root {
        --primary-color: #6366f1;
        --primary-dark: #4f46e5;
        --primary-light: #a5b4fc;
        --secondary-color: #10b981;
        --secondary-dark: #059669;
        --warning-color: #f59e0b;
        --warning-dark: #d97706;
        --error-color: #ef4444;
        --error-dark: #dc2626;
        --success-color: #10b981;
        --success-dark: #059669;
        
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        
        --bg-primary: #ffffff;
        --bg-secondary: #f9fafb;
        --bg-tertiary: #f3f4f6;
        --border-color: #e5e7eb;
        --border-light: #f3f4f6;
        
        --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.05);
        --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.1), 0 4px 6px -4px rgb(0 0 0 / 0.1);
        
        --radius-sm: 0.375rem;
        --radius-md: 0.5rem;
        --radius-lg: 0.75rem;
    }
    
    /* Base typography */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-header {
        font-size: 2.75rem;
        color: var(--text-primary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--primary-color), var(--secondary-color));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    .mode-header {
        font-size: 1.875rem;
        color: var(--text-primary);
        margin-bottom: 1.5rem;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Card components */
    .card-base {
        background-color: var(--bg-primary);
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        box-shadow: var(--shadow-sm);
        transition: all 0.2s ease-in-out;
    }
    
    .card-base:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .summary-box {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        padding: 2rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 2rem;
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
    }
    
    .summary-box h4 {
        color: var(--primary-color);
        margin-bottom: 1rem;
        font-weight: 600;
        font-size: 1.125rem;
    }
    
    .feature-card {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: var(--radius-lg);
        border: 1px solid var(--border-color);
        margin-bottom: 1rem;
        transition: all 0.2s ease-in-out;
        height: 100%;
    }
    
    .feature-card:hover {
        box-shadow: var(--shadow-md);
        transform: translateY(-2px);
        border-color: var(--primary-light);
    }
    
    .feature-card h4 {
        margin-bottom: 0.75rem;
        font-weight: 600;
        font-size: 1.125rem;
    }
    
    .feature-card p {
        color: var(--text-secondary);
        margin: 0;
        line-height: 1.5;
    }
    
    .feature-card-primary h4 { color: var(--primary-color); }
    .feature-card-secondary h4 { color: var(--secondary-color); }
    .feature-card-warning h4 { color: var(--warning-color); }
    
    /* Question and Answer boxes */
    .question-box {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        margin-bottom: 1rem;
        border: 1px solid #dbeafe;
        border-left: 4px solid var(--primary-color);
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
    }
    
    .question-box strong {
        color: var(--primary-color);
        font-weight: 600;
    }
    
    .answer-box {
        background: var(--bg-primary);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        border-left: 4px solid var(--secondary-color);
        margin-bottom: 1rem;
        border: 1px solid #d1fae5;
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
    }
    
    .answer-box strong {
        color: var(--secondary-color);
        font-weight: 600;
    }
    
    .evaluation-box {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        border-left: 4px solid var(--warning-color);
        margin-bottom: 1rem;
        border: 1px solid #fed7aa;
        color: var(--text-primary);
        box-shadow: var(--shadow-sm);
    }
    
    .evaluation-box h4 {
        color: var(--warning-dark);
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .evaluation-box strong {
        color: var(--warning-dark);
        font-weight: 600;
    }
    
    /* Score indicators */
    .score-excellent {
        color: #065f46;
        font-weight: 600;
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        padding: 0.375rem 0.75rem;
        border-radius: var(--radius-sm);
        border: 1px solid #6ee7b7;
    }
    
    .score-good {
        color: #92400e;
        font-weight: 600;
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        padding: 0.375rem 0.75rem;
        border-radius: var(--radius-sm);
        border: 1px solid #fcd34d;
    }
    
    .score-poor {
        color: #991b1b;
        font-weight: 600;
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        padding: 0.375rem 0.75rem;
        border-radius: var(--radius-sm);
        border: 1px solid #f87171;
    }
    
    .highlight {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        padding: 0.25rem 0.5rem;
        border-radius: var(--radius-sm);
        font-weight: 500;
        color: var(--warning-dark);
        border: 1px solid #fcd34d;
    }
    
    /* Improve general text visibility */
    .stMarkdown p {
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    .stMarkdown li {
        color: var(--text-primary);
        line-height: 1.6;
    }
    
    /* Button improvements */
    .stButton button {
        background: linear-gradient(135deg, var(--primary-color), var(--primary-dark));
        color: white;
        border: none;
        border-radius: var(--radius-md);
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
        transition: all 0.2s ease-in-out;
        box-shadow: var(--shadow-sm);
    }
    
    .stButton button:hover {
        background: linear-gradient(135deg, var(--primary-dark), #3730a3);
        box-shadow: var(--shadow-md);
        transform: translateY(-1px);
    }
    
    .stButton button:active {
        transform: translateY(0);
        box-shadow: var(--shadow-sm);
    }
    
    /* Input styling */
    .stTextInput input {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 0.75rem;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        transition: border-color 0.2s ease-in-out;
        background-color: var(--bg-primary);
    }
    
    .stTextInput input:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgb(99 102 241 / 0.1);
        outline: none;
    }
    
    .stTextArea textarea {
        border: 1px solid var(--border-color);
        border-radius: var(--radius-md);
        padding: 0.75rem;
        color: var(--text-primary);
        font-family: 'Inter', sans-serif;
        transition: border-color 0.2s ease-in-out;
        background-color: var(--bg-primary);
    }
    
    .stTextArea textarea:focus {
        border-color: var(--primary-color);
        box-shadow: 0 0 0 3px rgb(99 102 241 / 0.1);
        outline: none;
    }
    
    /* Sidebar improvements */
    .css-1d391kg {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, var(--bg-tertiary) 100%);
        border-right: 1px solid var(--border-color);
    }
    
    .css-1d391kg .stMarkdown {
        color: var(--text-primary);
    }
    
    .css-1d391kg h3 {
        color: var(--text-primary);
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: var(--text-primary);
        font-weight: 500;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: 2px dashed var(--border-color);
        border-radius: var(--radius-lg);
        background-color: var(--bg-secondary);
        transition: all 0.2s ease-in-out;
    }
    
    .stFileUploader [data-testid="stFileUploaderDropzone"]:hover {
        border-color: var(--primary-color);
        background-color: #f8fafc;
    }
    
    /* Alert improvements */
    .stAlert {
        border-radius: var(--radius-md);
        border: none;
        box-shadow: var(--shadow-sm);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, #d1fae5, #a7f3d0);
        color: #065f46;
        border-left: 4px solid var(--success-color);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fef3c7, #fde68a);
        color: #92400e;
        border-left: 4px solid var(--warning-color);
    }
    
    .stError {
        background: linear-gradient(135deg, #fee2e2, #fecaca);
        color: #991b1b;
        border-left: 4px solid var(--error-color);
    }
    
    /* Spinner */
    .stSpinner {
        color: var(--primary-color);
    }
    
    /* Responsive improvements */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        
        .mode-header {
            font-size: 1.5rem;
        }
        
        .feature-card {
            padding: 1rem;
        }
        
        .summary-box {
            padding: 1.5rem;
        }
    }
    
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        :root {
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --text-muted: #9ca3af;
            
            --bg-primary: #1f2937;
            --bg-secondary: #111827;
            --bg-tertiary: #0f172a;
            --border-color: #374151;
            --border-light: #4b5563;
        }
        
        .main-header {
            background: linear-gradient(135deg, #a5b4fc, #34d399);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        
        .summary-box {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-color: var(--border-color);
        }
        
        .feature-card {
            background-color: var(--bg-primary);
            border-color: var(--border-color);
        }
        
        .question-box {
            background-color: var(--bg-primary);
            border-color: #1e40af;
        }
        
        .answer-box {
            background-color: var(--bg-primary);
            border-color: #047857;
        }
        
        .evaluation-box {
            background: linear-gradient(135deg, #1e293b, #0f172a);
            border-color: #d97706;
        } 
        
        .stFileUploader [data-testid="stFileUploaderDropzone"] {
            background-color: var(--bg-secondary);
            border-color: var(--border-color);
        }
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-tertiary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border-color);
        border-radius: var(--radius-sm);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-muted);
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'document_uploaded' not in st.session_state:
        st.session_state.document_uploaded = False
    if 'summary' not in st.session_state:
        st.session_state.summary = ""
    if 'challenge_questions' not in st.session_state:
        st.session_state.challenge_questions = []
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_mode' not in st.session_state:
        st.session_state.current_mode = "upload"

def upload_document(file) -> Dict:
    """Upload document to backend"""
    try:
        files = {"file": (file.name, file.read(), file.type)}
        response = requests.post(f"{API_BASE_URL}/upload", files=files)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading document: {str(e)}")
        return None

def ask_question(question: str) -> Dict:
    """Ask a question to the assistant"""
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id
        }
        response = requests.post(f"{API_BASE_URL}/ask", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error asking question: {str(e)}")
        return None

def submit_challenge_answer(question_id: str, answer: str) -> Dict:
    """Submit answer to challenge question"""
    try:
        payload = {
            "question_id": question_id,
            "answer": answer,
            "session_id": st.session_state.session_id
        }
        response = requests.post(f"{API_BASE_URL}/challenge/answer", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error submitting answer: {str(e)}")
        return None

def get_conversation_history() -> Dict:
    """Get conversation history"""
    try:
        response = requests.get(f"{API_BASE_URL}/session/{st.session_state.session_id}/history")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting history: {str(e)}")
        return None

def format_score_color(score: int) -> str:
    """Format score with appropriate color"""
    if score >= 80:
        return f'<span class="score-excellent">{score}/100</span>'
    elif score >= 60:
        return f'<span class="score-good">{score}/100</span>'
    else:
        return f'<span class="score-poor">{score}/100</span>'

def render_upload_section():
    """Render document upload section"""
    st.markdown('<h1 class="main-header">🔍 Smart Research Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style="color: var(--text-secondary); font-size: 1.125rem; margin-bottom: 2.5rem; text-align: center; max-width: 600px; margin-left: auto; margin-right: auto;">
    Welcome to the Smart Research Assistant! This AI-powered tool helps you analyze documents through intelligent questioning and comprehension testing.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card feature-card-primary">
            <h4>🤔 Ask Anything</h4>
            <p>Get detailed answers to your questions with document-based justification and context</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card feature-card-secondary">
            <h4>🧠 Challenge Me</h4>
            <p>Test your understanding with AI-generated questions and receive detailed feedback</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card feature-card-warning">
            <h4>💭 Smart Memory</h4>
            <p>Maintains context across conversations for intelligent follow-up questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### 📁 Upload Your Document")
    st.markdown('<p style="color: var(--text-secondary); margin-bottom: 1.5rem;">Upload a research paper, report, or any structured document to get started with intelligent analysis.</p>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose a PDF or TXT file",
        type=['pdf', 'txt'],
        help="Upload a research paper, report, or any structured document"
    )
    
    if uploaded_file is not None:
        with st.spinner("🔄 Processing document... This may take a moment."):
            result = upload_document(uploaded_file)
            
            if result:
                st.session_state.session_id = result["session_id"]
                st.session_state.summary = result["summary"]
                st.session_state.challenge_questions = result["challenge_questions"]
                st.session_state.document_uploaded = True
                st.session_state.current_mode = "summary"
                st.success("✅ Document uploaded and processed successfully!")
                st.rerun()

def render_summary_section():
    """Render document summary section"""
    st.markdown('<h2 class="mode-header">📋 Document Summary</h2>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="summary-box">
        <h4>📄 Auto-Generated Summary</h4>
        <p style="line-height: 1.6; margin: 0;">{st.session_state.summary}</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1.5rem; font-weight: 600;">Choose Your Interaction Mode:</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🤔 Ask Anything", use_container_width=True):
            st.session_state.current_mode = "ask"
            st.rerun()
    
    with col2:
        if st.button("🧠 Challenge Me", use_container_width=True):
            st.session_state.current_mode = "challenge"
            st.rerun()


def render_ask_mode():
    """Render Ask Anything mode"""
    st.markdown('<h2 class="mode-header">🤔 Ask Anything</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: var(--text-secondary); font-size: 1.125rem; margin-bottom: 2rem; line-height: 1.6;">
    Ask any question about your document. The assistant will provide comprehensive answers based on the document content 
    with proper justification and references.
    </p>
    """, unsafe_allow_html=True)
    
    # Display conversation history
    if st.session_state.conversation_history:
        st.markdown('<h3 style="color: var(--text-primary); margin-bottom: 1.5rem; font-weight: 600;">💬 Conversation History</h3>', unsafe_allow_html=True)
        for i, item in enumerate(st.session_state.conversation_history):
            st.markdown(f"""
            <div class="question-box">
                <strong>Q{i+1}:</strong> {item['question']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="answer-box">
                <strong>Answer:</strong> {item['answer']}
            </div>
            """, unsafe_allow_html=True)
            
       
            if item.get('source_snippets'):
                with st.expander("📄 View Supporting Evidence", expanded=False):
                    for snippet in item['source_snippets']:
                        st.markdown(f"""
                        <div style="background: var(--bg-secondary); padding: 1rem; border-radius: var(--radius-md); margin-bottom: 0.5rem; border-left: 3px solid var(--primary-color);">
                            <span style="font-weight: 600; color: var(--primary-color); font-size: 0.875rem;">Page {snippet['page_number']}</span>
                            <p style="margin: 0.5rem 0 0 0; font-style: italic; color: var(--text-secondary);">"{snippet['content']}"</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    # Question input
    question = st.text_input(
        "💭 Ask your question:",
        placeholder="What is the main argument of this document?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        if st.button("Submit", use_container_width=True):
            if question:
                with st.spinner("🤔 Thinking..."):
                    result = ask_question(question)
                    if result:
                      
                        conversation_entry = {
                            'question': question,
                            'answer': result['answer']
                        }
                        
                       
                        if result.get('source_snippets'):
                            conversation_entry['source_snippets'] = [
    {
        'page_number': snippet.get('page_number') if isinstance(snippet, dict) else getattr(snippet, 'page_number', None),
        'content': snippet.get('content') if isinstance(snippet, dict) else getattr(snippet, 'content', '')
    }
    for snippet in result.get('source_snippets', [])
]

                        
                        st.session_state.conversation_history.append(conversation_entry)
                        st.rerun()
            else:
                st.warning("Please enter a question.")
    
    with col2:
        if st.button("🔄 New Conversation", use_container_width=True):
            st.session_state.conversation_history = []
            st.rerun()

def render_challenge_mode():
    """Render Challenge Me mode"""
    st.markdown('<h2 class="mode-header">🧠 Challenge Me</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <p style="color: var(--text-secondary); font-size: 1.125rem; margin-bottom: 2rem; line-height: 1.6;">
    Test your understanding with AI-generated questions. Answer them and receive detailed feedback 
    based on the document content with scoring and improvement suggestions.
    </p>
    """, unsafe_allow_html=True)
    
    if st.session_state.challenge_questions:
        for i, question_data in enumerate(st.session_state.challenge_questions):
            difficulty_emoji = {"easy": "🟢", "medium": "🟡", "hard": "🔴"}.get(question_data.get('difficulty', 'medium').lower(), "🟡")
            
            st.markdown(f"""
            <div class="question-box">
                <h4>Question {i+1} {difficulty_emoji} {question_data.get('difficulty', 'medium').title()}</h4>
                <p style="margin: 0; line-height: 1.6;">{question_data['question']}</p>
            </div>
            """, unsafe_allow_html=True)
            
            answer_key = f"answer_{question_data['id']}"
            user_answer = st.text_area(
                f"✍️ Your answer to Question {i+1}:",
                key=answer_key,
                height=120,
                placeholder="Type your detailed answer here..."
            )
            
            submit_key = f"submit_{question_data['id']}"
            evaluation_key = f"eval_{question_data['id']}"
            
            if st.button(f"Submit Answer {i+1}", key=submit_key):
                if user_answer:
                    with st.spinner("📊 Evaluating your answer..."):
                        result = submit_challenge_answer(question_data['id'], user_answer)
                        if result:
                            st.session_state[evaluation_key] = result['evaluation']
                            st.rerun()
                else:
                    st.warning("Please provide an answer.")
            
            # Display evaluation if available
            if evaluation_key in st.session_state:
                eval_data = st.session_state[evaluation_key]
                score_html = format_score_color(eval_data['score'])
                
                st.markdown(f"""
                <div class="evaluation-box">
                    <h4>📊 Evaluation Results</h4>
                    <p><strong>Score:</strong> {score_html}</p>
                    <p><strong>Feedback:</strong> {eval_data['feedback']}</p>
                    <p><strong>Document Reference:</strong> {eval_data['document_reference']}</p>
                    <p><strong>Suggestions:</strong> {eval_data['suggestions']}</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")

def render_sidebar():
    """Render sidebar with navigation and controls"""
    st.sidebar.markdown('<h3 style="color: var(--text-primary); font-weight: 600;">🎯 Navigation</h3>', unsafe_allow_html=True)
    
    if st.session_state.document_uploaded:
        if st.sidebar.button("📋 Summary", use_container_width=True):
            st.session_state.current_mode = "summary"
            st.rerun()
        
        if st.sidebar.button("🤔 Ask Anything", use_container_width=True):
            st.session_state.current_mode = "ask"
            st.rerun()
        
        if st.sidebar.button("🧠 Challenge Me", use_container_width=True):
            st.session_state.current_mode = "challenge"
            st.rerun()
        
        st.sidebar.markdown("---")
        
        if st.sidebar.button("🔄 Upload New Document", use_container_width=True):
            # Reset session state
            st.session_state.session_id = None
            st.session_state.document_uploaded = False
            st.session_state.summary = ""
            st.session_state.challenge_questions = []
            st.session_state.conversation_history = []
            st.session_state.current_mode = "upload"
            st.rerun()
    
    # App info
    st.sidebar.markdown('<h3 style="color: var(--text-primary); font-weight: 600;">ℹ️ Technology Stack</h3>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="color: var(--text-secondary); line-height: 1.6;">
    <strong>🔧 Backend:</strong><br>
    • LangChain for document processing<br>
    • LangGraph for workflow management<br>
    • Gemini api for AI reasoning<br>
    • FAISS for vector search<br>
    • FastAPI for backend API<br><br>
    
    <strong>🎨 Frontend:</strong><br>
    • Streamlit for web interface<br>
    • Custom CSS for modern design<br>
    • Responsive layout support
    </div>
    """, unsafe_allow_html=True)
    
    # Features
    st.sidebar.markdown('<h3 style="color: var(--text-primary); font-weight: 600;">✨ Key Features</h3>', unsafe_allow_html=True)
    st.sidebar.markdown("""
    <div style="color: var(--text-secondary); line-height: 1.6;">
    • 📄 PDF & TXT document support<br>
    • 🔍 Contextual question answering<br>
    • 🧠 Logic-based challenge questions<br>
    • 💭 Conversation memory & follow-ups<br>
    • 📊 Detailed answer evaluation<br>
    • 🎯 Document-based references<br>
    • 📱 Mobile-responsive design
    </div>
    """, unsafe_allow_html=True)

def main():
    """Main application function"""
    initialize_session_state()
    render_sidebar()
    
    # Main content area
    if st.session_state.current_mode == "upload":
        render_upload_section()
    elif st.session_state.current_mode == "summary":
        render_summary_section()
    elif st.session_state.current_mode == "ask":
        render_ask_mode()
    elif st.session_state.current_mode == "challenge":
        render_challenge_mode()

if __name__ == "__main__":
    main()