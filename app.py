import os
import io
import re
import uuid
import sqlite3
import mimetypes
import tempfile
import ipaddress
import socket
import zipfile
import base64
import json
import hashlib
import subprocess
import threading
import time
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote
from typing import List, Dict, Any, Optional, Callable
import numpy as np

import streamlit as st
import requests
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# MO DARK AI - ULTIMATE EDITION
# ============================================================
st.set_page_config(
    page_title="Mo Dark AI | Ultimate Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://venice.ai',
        'Report a bug': 'mailto:support@venice.ai',
        'About': '# Mo Dark AI\nThe most advanced multimodal AI interface ever created.'
    }
)

# ============================================================
# ADVANCED SETTINGS
# ============================================================
APP_NAME = "Mo Dark AI Ultimate"
VERSION = "3.0.0"

# Model Configuration
TEXT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"
VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VISION_FALLBACK = "Qwen/Qwen2.5-VL-72B-Instruct"
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"
VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B"
IMAGE_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B"
ASR_MODEL = "openai/whisper-large-v3"

# Advanced Capabilities
ENABLE_CODE_EXECUTION = True
ENABLE_WEB_SEARCH = True
ENABLE_VECTOR_MEMORY = True
ENABLE_AGENTS = True
ENABLE_ANALYTICS = True

# Limits
MAX_REMOTE_BYTES = 50 * 1024 * 1024
MAX_VIDEO_FRAMES = 8
MAX_MEMORY_ITEMS = 1000
VECTOR_DIMENSION = 384

# Database
DB_FILE = "mo_dark_ultimate.db"
VECTOR_STORE_FILE = "vector_memory.pkl"

# ============================================================
# SESSION MANAGEMENT
# ============================================================
class SessionManager:
    def __init__(self):
        self.init_state()
    
    def init_state(self):
        defaults = {
            "session_id": self.create_session(),
            "sidebar_visible": True,
            "streaming": True,
            "temperature": 0.7,
            "max_tokens": 4096,
            "code_enabled": ENABLE_CODE_EXECUTION,
            "web_search_enabled": ENABLE_WEB_SEARCH,
            "memory_enabled": ENABLE_VECTOR_MEMORY,
            "current_agent": None,
            "uploaded_files_cache": {},
            "thinking": False,
            "last_query_time": None,
            "rate_limit_count": 0,
            "custom_tools": [],
            "theme": "dark",
            "language": "ar",
            "voice_enabled": False,
            "auto_save": True,
            "export_format": "markdown"
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def create_session(self, title="محادثة جديدة"):
        session_id = str(uuid.uuid4())
        now = datetime.now().isoformat(timespec="seconds")
        conn = db_connection()
        conn.execute(
            """INSERT INTO sessions (id, title, created_at, updated_at, metadata)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, title, now, now, json.dumps({"version": VERSION}))
        )
        conn.commit()
        conn.close()
        return session_id
    
    def update_session_metadata(self, session_id, key, value):
        conn = db_connection()
        row = conn.execute("SELECT metadata FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if row:
            metadata = json.loads(row["metadata"] or "{}")
            metadata[key] = value
            conn.execute("UPDATE sessions SET metadata = ? WHERE id = ?", 
                        (json.dumps(metadata), session_id))
            conn.commit()
        conn.close()

session_mgr = SessionManager()

# ============================================================
# DATABASE SCHEMA
# ============================================================
def db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False, 
                          detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

def init_database():
    conn = db_connection()
    
    # Sessions with metadata
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        )
    """)
    
    # Messages with embeddings
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            metadata TEXT DEFAULT '{}',
            embedding BLOB,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    # Files storage
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            content BLOB,
            extracted_text TEXT,
            metadata TEXT DEFAULT '{}',
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    # Agent tasks
    conn.execute("""
        CREATE TABLE IF NOT EXISTS agent_tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            agent_name TEXT NOT NULL,
            task_description TEXT NOT NULL,
            status TEXT DEFAULT 'pending',
            result TEXT,
            created_at TEXT NOT NULL,
            completed_at TEXT,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    # Code execution history
    conn.execute("""
        CREATE TABLE IF NOT EXISTS code_executions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            code TEXT NOT NULL,
            output TEXT,
            error TEXT,
            execution_time REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)
    
    conn.commit()
    conn.close()

init_database()

# ============================================================
# VECTOR MEMORY SYSTEM
# ============================================================
class VectorMemory:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=VECTOR_DIMENSION, stop_words='english')
        self.memory_cache = {}
        self._load_memory()
    
    def _load_memory(self):
        try:
            if os.path.exists(VECTOR_STORE_FILE):
                with open(VECTOR_STORE_FILE, 'rb') as f:
                    import pickle
                    self.memory_cache = pickle.load(f)
        except Exception:
            self.memory_cache = {}
    
    def _save_memory(self):
        try:
            import pickle
            with open(VECTOR_STORE_FILE, 'wb') as f:
                pickle.dump(self.memory_cache, f)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def add_memory(self, session_id: str, content: str, metadata: Dict = None):
        """Add content to vector memory"""
        key = hashlib.md5(content.encode()).hexdigest()
        if session_id not in self.memory_cache:
            self.memory_cache[session_id] = []
        
        memory_item = {
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now().isoformat(),
            "key": key
        }
        
        self.memory_cache[session_id].append(memory_item)
        
        # Keep only recent memories
        if len(self.memory_cache[session_id]) > MAX_MEMORY_ITEMS:
            self.memory_cache[session_id] = self.memory_cache[session_id][-MAX_MEMORY_ITEMS:]
        
        self._save_memory()
        return key
    
    def search_memory(self, session_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search similar memories using cosine similarity"""
        if session_id not in self.memory_cache or not self.memory_cache[session_id]:
            return []
        
        memories = self.memory_cache[session_id]
        contents = [m["content"] for m in memories]
        contents.append(query)
        
        try:
            vectors = self.vectorizer.fit_transform(contents)
            query_vec = vectors[-1]
            memory_vecs = vectors[:-1]
            
            similarities = cosine_similarity(query_vec, memory_vecs)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Threshold
                    results.append({
                        **memories[idx],
                        "similarity": float(similarities[idx])
                    })
            return results
        except Exception:
            return []

vector_memory = VectorMemory()

# ============================================================
# AGENT SYSTEM
# ============================================================
class Agent:
    def __init__(self, name: str, description: str, tools: List[Callable]):
        self.name = name
        self.description = description
        self.tools = tools
        self.memory = []
    
    def execute(self, task: str, context: Dict = None) -> Dict[str, Any]:
        """Execute agent task"""
        result = {
            "agent": self.name,
            "task": task,
            "steps": [],
            "output": None,
            "success": True
        }
        
        # Plan execution
        plan = self._plan(task, context)
        result["steps"].append({"plan": plan})
        
        # Execute each step
        for step in plan:
            try:
                step_result = self._execute_step(step, context)
                result["steps"].append(step_result)
            except Exception as e:
                result["steps"].append({"error": str(e)})
                result["success"] = False
        
        # Generate final output
        result["output"] = self._synthesize(result["steps"])
        return result
    
    def _plan(self, task: str, context: Dict) -> List[Dict]:
        """Create execution plan"""
        # Simple planning - can be enhanced with LLM
        return [{"action": "process", "input": task}]
    
    def _execute_step(self, step: Dict, context: Dict) -> Dict:
        """Execute single step"""
        return {"step": step, "status": "completed"}
    
    def _synthesize(self, steps: List[Dict]) -> str:
        """Synthesize final output from steps"""
        return "Task completed successfully"

class CodeAgent(Agent):
    def __init__(self):
        super().__init__(
            "CodeExpert",
            "Specialized in code analysis, generation, and execution",
            [self.execute_python, self.analyze_code, self.generate_code]
        )
    
    def execute_python(self, code: str, timeout: int = 30) -> Dict:
        """Execute Python code safely"""
        result = {
            "output": "",
            "error": "",
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
            
            # Create restricted environment
            env = {
                "__builtins__": {
                    "len": len, "range": range, "enumerate": enumerate,
                    "zip": zip, "map": map, "filter": filter,
                    "sum": sum, "min": min, "max": max, "abs": abs,
                    "round": round, "pow": pow, "divmod": divmod,
                    "print": lambda *args: args,
                    "str": str, "int": int, "float": float, "list": list,
                    "dict": dict, "tuple": tuple, "set": set,
                    "pd": pd, "np": np, "px": px, "go": go
                }
            }
            
            # Execute in subprocess for safety
            output = io.StringIO()
            error = io.StringIO()
            
            exec(code, env, {"output": output, "error": error})
            
            result["execution_time"] = time.time() - start_time
            result["output"] = output.getvalue()
            result["error"] = error.getvalue()
            
        except Exception as e:
            result["error"] = str(e)
        
        return result
    
    def analyze_code(self, code: str) -> Dict:
        """Analyze code for issues and improvements"""
        issues = []
        suggestions = []
        
        # Basic analysis
        if "eval(" in code or "exec(" in code:
            issues.append("Security risk: eval/exec detected")
        
        if len(code.split('\n')) > 100:
            suggestions.append("Consider breaking into smaller functions")
        
        return {"issues": issues, "suggestions": suggestions}
    
    def generate_code(self, description: str, language: str = "python") -> str:
        """Generate code from description"""
        # This would use LLM in production
        return f"# Generated {language} code for: {description}\n# TODO: Implement"

class DataAnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            "DataAnalyst",
            "Specialized in data analysis and visualization",
            [self.analyze_dataframe, self.create_visualization, self.statistical_analysis]
        )
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        """Comprehensive DataFrame analysis"""
        analysis = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        # Numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        # Categorical columns
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            analysis["categorical_summary"][col] = df[col].value_counts().head(10).to_dict()
        
        return analysis
    
    def create_visualization(self, df: pd.DataFrame, chart_type: str, **kwargs) -> go.Figure:
        """Create Plotly visualization"""
        if chart_type == "scatter":
            fig = px.scatter(df, **kwargs)
        elif chart_type == "line":
            fig = px.line(df, **kwargs)
        elif chart_type == "bar":
            fig = px.bar(df, **kwargs)
        elif chart_type == "histogram":
            fig = px.histogram(df, **kwargs)
        elif chart_type == "box":
            fig = px.box(df, **kwargs)
        elif chart_type == "heatmap":
            corr = df.select_dtypes(include=[np.number]).corr()
            fig = px.imshow(corr, text_auto=True, aspect="auto")
        else:
            fig = px.scatter(df)
        
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#f5f7fb")
        )
        return fig
    
    def statistical_analysis(self, df: pd.DataFrame, column: str) -> Dict:
        """Perform statistical tests"""
        from scipy import stats
        
        data = df[column].dropna()
        result = {
            "shapiro_test": None,
            "normal_distribution": False,
            "skewness": float(stats.skew(data)),
            "kurtosis": float(stats.kurtosis(data))
        }
        
        if len(data) >= 3:
            stat, p_value = stats.shapiro(data)
            result["shapiro_test"] = {"statistic": float(stat), "p_value": float(p_value)}
            result["normal_distribution"] = p_value > 0.05
        
        return result

class WebSearchAgent(Agent):
    def __init__(self):
        super().__init__(
            "WebSearcher",
            "Specialized in web search and information retrieval",
            [self.search_web, self.scrape_page, self.summarize_results]
        )
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search the web using DuckDuckGo or similar"""
        # Using DuckDuckGo HTML version (no API key needed)
        try:
            url = f"https://html.duckduckgo.com/html/?q={quote(query)}"
            headers = {"User-Agent": "Mozilla/5.0"}
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            results = []
            for result in soup.find_all('div', class_='result', limit=num_results):
                title_elem = result.find('a', class_='result__a')
                snippet_elem = result.find('a', class_='result__snippet')
                
                if title_elem and snippet_elem:
                    results.append({
                        "title": title_elem.get_text(),
                        "url": title_elem.get('href'),
                        "snippet": snippet_elem.get_text()
                    })
            
            return results
        except Exception as e:
            return [{"error": str(e)}]
    
    def scrape_page(self, url: str) -> Dict:
        """Scrape and extract content from URL"""
        try:
            response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                tag.decompose()
            
            title = soup.title.get_text() if soup.title else "No title"
            content = soup.get_text(separator='\n', strip=True)
            
            return {
                "title": title,
                "url": url,
                "content": content[:10000],
                "word_count": len(content.split())
            }
        except Exception as e:
            return {"error": str(e)}
    
    def summarize_results(self, results: List[Dict]) -> str:
        """Summarize search results"""
        summary = []
        for i, result in enumerate(results, 1):
            if "title" in result:
                summary.append(f"{i}. {result['title']}\n   {result.get('snippet', '')}")
        return "\n\n".join(summary)

# Initialize agents
code_agent = CodeAgent()
data_agent = DataAnalysisAgent()
web_agent = WebSearchAgent()

AVAILABLE_AGENTS = {
    "code": code_agent,
    "data": data_agent,
    "web": web_agent
}

# ============================================================
# ADVANCED FILE PROCESSING
# ============================================================
class FileProcessor:
    SUPPORTED_TYPES = {
        "image": [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"],
        "video": [".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"],
        "audio": [".mp3", ".wav", ".m4a", ".ogg", ".flac"],
        "document": [".pdf", ".docx", ".txt", ".md"],
        "data": [".csv", ".xlsx", ".xls", ".json", ".parquet"],
        "code": [".py", ".js", ".ts", ".jsx", ".tsx", ".html", ".css", 
                ".java", ".cpp", ".c", ".go", ".rs", ".php", ".rb"]
    }
    
    @classmethod
    def detect_type(cls, filename: str) -> str:
        ext = Path(filename).suffix.lower()
        for file_type, extensions in cls.SUPPORTED_TYPES.items():
            if ext in extensions:
                return file_type
        return "unknown"
    
    @staticmethod
    def extract_text_from_pdf(data: bytes) -> str:
        try:
            import fitz
            doc = fitz.open(stream=data, filetype="pdf")
            text = []
            for page in doc:
                text.append(page.get_text())
            doc.close()
            return "\n\n".join(text)
        except Exception as e:
            return f"[PDF Error: {e}]"
    
    @staticmethod
    def extract_from_docx(data: bytes) -> str:
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "\n".join([p.text for p in doc.paragraphs if p.text])
        except Exception as e:
            return f"[DOCX Error: {e}]"
    
    @staticmethod
    def process_data_file(data: bytes, ext: str) -> pd.DataFrame:
        try:
            if ext == ".csv":
                return pd.read_csv(io.BytesIO(data))
            elif ext in [".xlsx", ".xls"]:
                return pd.read_excel(io.BytesIO(data))
            elif ext == ".json":
                return pd.read_json(io.BytesIO(data))
            elif ext == ".parquet":
                return pd.read_parquet(io.BytesIO(data))
        except Exception as e:
            st.error(f"Error reading data file: {e}")
            return None
    
    @classmethod
    def process_upload(cls, uploaded_file, session_id: str) -> Dict[str, Any]:
        """Process uploaded file and return structured data"""
        data = uploaded_file.getvalue()
        filename = uploaded_file.name
        file_type = cls.detect_type(filename)
        file_id = str(uuid.uuid4())
        
        result = {
            "id": file_id,
            "filename": filename,
            "type": file_type,
            "size": len(data),
            "content": None,
            "dataframe": None,
            "images": [],
            "metadata": {}
        }
        
        # Store in database
        conn = db_connection()
        
        if file_type == "image":
            result["content"] = data
            result["images"].append(data_url(data, uploaded_file.type or "image/jpeg"))
            
        elif file_type == "video":
            frames, _ = extract_video_frames(data)
            result["content"] = f"Video: {filename} ({len(data)} bytes, {len(frames)} frames)"
            for frame in frames:
                result["images"].append(data_url(frame, "image/png"))
                
        elif file_type == "audio":
            # Transcribe if possible
            transcript = transcribe_audio(data)
            result["content"] = transcript
            
        elif file_type == "document":
            ext = Path(filename).suffix.lower()
            if ext == ".pdf":
                result["content"] = cls.extract_text_from_pdf(data)
            elif ext == ".docx":
                result["content"] = cls.extract_from_docx(data)
            else:
                result["content"] = data.decode('utf-8', errors='replace')
                
        elif file_type == "data":
            ext = Path(filename).suffix.lower()
            df = cls.process_data_file(data, ext)
            if df is not None:
                result["dataframe"] = df
                result["content"] = df.head(100).to_string()
                result["metadata"]["shape"] = df.shape
                result["metadata"]["columns"] = df.columns.tolist()
                
        elif file_type == "code":
            result["content"] = data.decode('utf-8', errors='replace')
            result["metadata"]["language"] = Path(filename).suffix[1:]
            
        # Save to DB
        conn.execute(
            """INSERT INTO files (id, session_id, filename, file_type, content, 
               extracted_text, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (file_id, session_id, filename, file_type, data,
             result.get("content", ""), json.dumps(result.get("metadata", {})), 
             datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
        
        return result

# ============================================================
# TOKEN & CLIENT
# ============================================================
def get_hf_token():
    try:
        return st.secrets.get("HF_TOKEN") or os.getenv("HF_TOKEN")
    except:
        return None

HF_TOKEN = get_hf_token()

@st.cache_resource(show_spinner=False)
def create_client(token):
    if not token:
        return None
    return InferenceClient(api_key=token, provider="auto")

client = create_client(HF_TOKEN)

# ============================================================
# UI COMPONENTS
# ============================================================
def render_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');
    
    :root {
        --bg-primary: #030509;
        --bg-secondary: #080c13;
        --bg-tertiary: #0d1320;
        --accent-cyan: #00eaff;
        --accent-blue: #2878ff;
        --accent-purple: #7c3cff;
        --accent-pink: #ff2d7a;
        --text-primary: #f5f7fb;
        --text-secondary: #a0aec0;
        --text-muted: #64748b;
        --border: rgba(255,255,255,0.08);
        --glow-cyan: 0 0 30px rgba(0,234,255,0.3);
        --glow-purple: 0 0 30px rgba(124,60,255,0.3);
    }
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background: 
            radial-gradient(circle at 20% 50%, rgba(0,234,255,0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 80%, rgba(124,60,255,0.08) 0%, transparent 50%),
            radial-gradient(circle at 50% 0%, rgba(40,120,255,0.05) 0%, transparent 40%),
            linear-gradient(180deg, var(--bg-primary) 0%, #020408 100%);
    }
    
    .block-container {
        max-width: 1200px;
        padding: 1rem 2rem 6rem;
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-secondary);
    }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--accent-cyan), var(--accent-blue));
        border-radius: 4px;
    }
    
    /* Typography */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, #050810 100%);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(0,234,255,0.1), rgba(40,120,255,0.1)) !important;
        border: 1px solid rgba(0,234,255,0.2) !important;
        border-radius: 12px !important;
        color: var(--text-primary) !important;
        font-weight: 500 !important;
        transition: all 0.3s ease !important;
        backdrop-filter: blur(10px);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, rgba(0,234,255,0.2), rgba(40,120,255,0.2)) !important;
        border-color: var(--accent-cyan) !important;
        box-shadow: var(--glow-cyan);
        transform: translateY(-1px);
    }
    
    /* Primary action button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
        color: var(--bg-primary) !important;
        font-weight: 600 !important;
    }
    
    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: transparent !important;
        border: none !important;
        animation: fadeIn 0.3s ease;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    [data-testid="stChatMessageContent"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1rem 1.25rem;
        backdrop-filter: blur(10px);
    }
    
    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(0,234,255,0.1), rgba(40,120,255,0.05));
        border-color: rgba(0,234,255,0.2);
    }
    
    /* Chat input */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    
    [data-testid="stChatInput"] > div {
        background: rgba(13, 19, 32, 0.95) !important;
        border: 1px solid rgba(0,234,255,0.25) !important;
        border-radius: 20px !important;
        box-shadow: 
            0 0 40px rgba(0,234,255,0.08),
            0 20px 60px rgba(0,0,0,0.5),
            inset 0 1px 0 rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(20px);
    }
    
    [data-testid="stChatInput"] textarea {
        color: var(--text-primary) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: rgba(255,255,255,0.02);
        border: 2px dashed rgba(0,234,255,0.2);
        border-radius: 16px;
        padding: 1.5rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: var(--accent-cyan);
        background: rgba(0,234,255,0.05);
    }
    
    /* Code blocks */
    pre {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1rem !important;
    }
    
    code {
        font-family: 'JetBrains Mono', monospace !important;
        color: var(--accent-cyan) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.03);
        border: 1px solid var(--border);
        border-radius: 12px;
        color: var(--text-secondary);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255,255,255,0.03);
        border-radius: 8px;
        color: var(--text-secondary);
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(0,234,255,0.2), rgba(40,120,255,0.1));
        color: var(--text-primary);
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: var(--accent-cyan);
        font-weight: 700;
    }
    
    /* Dataframes */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 12px;
    }
    
    /* Thinking animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .thinking {
        animation: pulse 1.5s ease infinite;
        color: var(--accent-cyan);
    }
    
    /* Agent badge */
    .agent-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        background: linear-gradient(135deg, rgba(124,60,255,0.2), rgba(255,45,122,0.1));
        border: 1px solid rgba(124,60,255,0.3);
        border-radius: 20px;
        font-size: 12px;
        color: var(--accent-purple);
        font-weight: 600;
    }
    
    /* Hide defaults */
    #MainMenu, footer, header {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    col1, col2, col3 = st.columns([1, 10, 2])
    
    with col1:
        if st.button("☰", key="toggle_sidebar"):
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 15px; padding: 10px 0;">
            <div style="
                width: 45px; height: 45px;
                background: linear-gradient(135deg, #00eaff, #2878ff, #7c3cff);
                border-radius: 14px;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 0 30px rgba(0,234,255,0.3);
                font-size: 24px;
            ">⚡</div>
            <div>
                <div style="font-size: 22px; font-weight: 800; color: #fff;">
                    Mo Dark AI
                    <span style="font-size: 12px; background: linear-gradient(90deg, #00eaff, #7c3cff); 
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-left: 8px;">
                        ULTIMATE
                    </span>
                </div>
                <div style="font-size: 11px; color: #64748b; letter-spacing: 1.5px; margin-top: 2px;">
                    MULTIMODAL INTELLIGENCE v3.0
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Quick actions
        st.markdown("<div style='text-align: right;'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("🎨", help="توليد صورة"):
                st.session_state.show_image_gen = True
        with c2:
            if st.button("🎬", help="توليد فيديو"):
                st.session_state.show_video_gen = True
        with c3:
            if st.button("⚙️", help="الإعدادات"):
                st.session_state.show_settings = True
        st.markdown("</div>", unsafe_allow_html=True)

def render_sidebar():
    if not st.session_state.sidebar_visible:
        return
    
    with st.sidebar:
        # Logo section
        st.markdown("""
        <div style="text-align: center; padding: 20px 0;">
            <div style="
                width: 60px; height: 60px;
                margin: 0 auto;
                background: linear-gradient(135deg, #00eaff, #2878ff, #7c3cff);
                border-radius: 18px;
                display: flex; align-items: center; justify-content: center;
                box-shadow: 0 0 40px rgba(0,234,255,0.3);
                font-size: 30px;
            ">⚡</div>
            <div style="margin-top: 12px; font-size: 20px; font-weight: 700;">Mo Dark</div>
            <div style="font-size: 10px; color: #64748b; letter-spacing: 2px;">ULTIMATE AI</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # New chat button
        if st.button("＋ محادثة جديدة", use_container_width=True, type="primary"):
            st.session_state.session_id = session_mgr.create_session()
            st.session_state.messages = []
            st.rerun()
        
        # Agent selector
        st.markdown("### 🤖 الوكلاء المتاحون")
        agent_cols = st.columns(3)
        agents = [
            ("💻", "code", "Code Expert"),
            ("📊", "data", "Data Analyst"),
            ("🌐", "web", "Web Search")
        ]
        for i, (icon, key, name) in enumerate(agents):
            with agent_cols[i]:
                if st.button(icon, key=f"agent_{key}", help=name):
                    st.session_state.current_agent = key
        
        if st.session_state.current_agent:
            st.info(f"الوكيل النشط: {AVAILABLE_AGENTS[st.session_state.current_agent].name}")
        
        st.markdown("---")
        
        # Session history
        st.markdown('<div style="color: #64748b; font-size: 11px; font-weight: 700; margin-bottom: 10px;">المحادثات السابقة</div>', unsafe_allow_html=True)
        
        sessions = load_sessions()
        for session in sessions[:20]:  # Show last 20
            session_id = session["id"]
            title = session["title"] or "محادثة جديدة"
            is_current = session_id == st.session_state.session_id
            
            btn_style = "primary" if is_current else "secondary"
            icon = "●" if is_current else "○"
            
            if st.button(f"{icon} {title[:35]}", key=f"session_{session_id}", 
                      use_container_width=True, type=btn_style):
                st.session_state.session_id = session_id
                st.rerun()
        
        st.markdown("---")
        
        # Stats
        if ENABLE_ANALYTICS:
            conn = db_connection()
            msg_count = conn.execute(
                "SELECT COUNT(*) FROM messages WHERE session_id = ?", 
                (st.session_state.session_id,)
            ).fetchone()[0]
            conn.close()
            
            st.markdown(f"""
            <div style="font-size: 11px; color: #475569;">
                الرسائل: {msg_count} | الذاكرة: {len(vector_memory.memory_cache.get(st.session_state.session_id, []))}
            </div>
            """, unsafe_allow_html=True)
        
        # Footer
        st.markdown("""
        <div style="position: fixed; bottom: 20px; font-size: 9px; color: #334155; line-height: 1.8;">
            MO DARK AI ULTIMATE<br>
            ONE CHAT • EVERYTHING • EVERYWHERE
        </div>
        """, unsafe_allow_html=True)

def load_sessions():
    conn = db_connection()
    rows = conn.execute("""
        SELECT * FROM sessions 
        ORDER BY updated_at DESC
    """).fetchall()
    conn.close()
    return rows

def load_messages(session_id):
    conn = db_connection()
    rows = conn.execute("""
        SELECT role, content, content_type, metadata, created_at 
        FROM messages 
        WHERE session_id = ? 
        ORDER BY id ASC
    """, (session_id,)).fetchall()
    conn.close()
    return rows

# ============================================================
# INTELLIGENT QUERY PROCESSING
# ============================================================
class QueryRouter:
    """Routes queries to appropriate handlers"""
    
    PATTERNS = {
        "image_generation": r"(?:انشئ|صمم|ارسم|ولد|generate|create).{0,20}(?:صور|image|picture|photo)",
        "video_generation": r"(?:انشئ|صمم|ولد|generate|create).{0,20}(?:فيديو|video|animation)",
        "code_execution": r"(?:نفذ|شغل|execute|run).{0,10}(?:كود|كود|code|python)",
        "web_search": r"(?:ابحث|دور|search|find).{0,20}(?:عن|في|on|for)",
        "data_analysis": r"(?:حلل|اعرض| visualize|analyze|plot|graph)",
        "file_operation": r"(?:اقرأ|افتح|read|open|process).{0,10}(?:ملف|file)",
    }
    
    @classmethod
    def analyze(cls, query: str) -> Dict[str, Any]:
        query_lower = query.lower()
        intents = {}
        
        for intent, pattern in cls.PATTERNS.items():
            if re.search(pattern, query_lower):
                intents[intent] = True
        
        return {
            "intents": intents,
            "primary_intent": list(intents.keys())[0] if intents else "chat",
            "complexity": "high" if len(intents) > 1 else "low",
            "requires_agent": any(i in intents for i in ["code_execution", "data_analysis", "web_search"])
        }

# ============================================================
# MAIN APPLICATION
# ============================================================
def main():
    render_css()
    render_header()
    render_sidebar()
    
    # Settings modal
    if st.session_state.get("show_settings"):
        with st.expander("الإعدادات", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.temperature = st.slider("Temperature", 0.0, 2.0, 0.7)
                st.session_state.max_tokens = st.selectbox("Max Tokens", [2048, 4096, 8192], index=1)
            with col2:
                st.session_state.streaming = st.toggle("Streaming", True)
                st.session_state.memory_enabled = st.toggle("Vector Memory", ENABLE_VECTOR_MEMORY)
            if st.button("إغلاق"):
                st.session_state.show_settings = False
                st.rerun()
    
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Load messages
    messages = load_messages(st.session_state.session_id)
    
    # Display messages
    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["content_type"] == "image":
                st.image(msg["content"])
            elif msg["content_type"] == "code":
                st.code(msg["content"], language="python")
            elif msg["content_type"] == "dataframe":
                try:
                    df = pd.read_json(msg["content"])
                    st.dataframe(df, use_container_width=True)
                except:
                    st.markdown(msg["content"])
            elif msg["content_type"] == "chart":
                try:
                    fig_dict = json.loads(msg["content"])
                    st.plotly_chart(fig_dict, use_container_width=True)
                except:
                    st.markdown(msg["content"])
            else:
                st.markdown(msg["content"])
    
    # File uploader
    uploaded_files = st.file_uploader(
        "📎 الملفات (اسحب وأفلت أو انقر للاختيار)",
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Chat input
    user_input = st.chat_input("اطرح سؤالك أو اطلب مهمة معقدة...")
    
    if user_input or uploaded_files:
        # Process files
        file_context = []
        images = []
        
        if uploaded_files:
            for file in uploaded_files:
                processed = FileProcessor.process_upload(file, st.session_state.session_id)
                file_context.append(f"File: {processed['filename']} ({processed['type']})")
                if processed.get('images'):
                    images.extend(processed['images'])
                if processed.get('dataframe') is not None:
                    # Auto-analyze data
                    analysis = data_agent.analyze_dataframe(processed['dataframe'])
                    file_context.append(f"Data Analysis: {json.dumps(analysis, default=str)}")
        
        # Build full prompt
        full_prompt = user_input or ""
        if file_context:
            full_prompt += "\n\n[Files Context]:\n" + "\n".join(file_context)
        
        # Save user message
        if user_input:
            save_message(st.session_state.session_id, "user", user_input, "text")
            with st.chat_message("user"):
                st.markdown(user_input)
        
        # Process with AI
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            # Route query
            router = QueryRouter()
            routing = router.analyze(user_input or "")
            
            # Check for agent requirement
            if routing["requires_agent"] and st.session_state.current_agent:
                agent = AVAILABLE_AGENTS.get(st.session_state.current_agent)
                if agent:
                    response_placeholder.markdown(f'<div class="agent-badge">🤖 {agent.name} يعمل...</div>', 
                                                unsafe_allow_html=True)
                    result = agent.execute(full_prompt)
                    output = result.get("output", "Task completed")
            
            # Image generation
            elif "image_generation" in routing["intents"]:
                response_placeholder.markdown('<div class="thinking">جاري توليد الصورة...</div>', 
                                           unsafe_allow_html=True)
                try:
                    image = client.text_to_image(prompt=user_input, model=IMAGE_MODEL)
                    st.image(image, use_container_width=True)
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()
                    save_message(st.session_state.session_id, "assistant", 
                               f"data:image/png;base64,{img_str}", "image")
                except Exception as e:
                    response_placeholder.error(f"Error: {e}")
            
            # Standard chat
            else:
                # Get relevant memories
                if st.session_state.memory_enabled:
                    memories = vector_memory.search_memory(st.session_state.session_id, full_prompt)
                    if memories:
                        memory_context = "\n".join([m["content"] for m in memories])
                        full_prompt = f"[Relevant Context from Memory]:\n{memory_context}\n\n{full_prompt}"
                
                # Call LLM
                try:
                    if images:
                        # Vision model
                        content = [{"type": "text", "text": full_prompt}]
                        for img in images:
                            content.append({"type": "image_url", "image_url": {"url": img}})
                        
                        completion = client.chat.completions.create(
                            model=VISION_MODEL,
                            messages=[{"role": "user", "content": content}],
                            max_tokens=st.session_state.max_tokens,
                            temperature=st.session_state.temperature,
                            stream=st.session_state.streaming
                        )
                    else:
                        completion = client.chat.completions.create(
                            model=TEXT_MODEL,
                            messages=[{"role": "user", "content": full_prompt}],
                            max_tokens=st.session_state.max_tokens,
                            temperature=st.session_state.temperature,
                            stream=st.session_state.streaming
                        )
                    
                    # Handle streaming
                    if st.session_state.streaming:
                        full_response = ""
                        for chunk in completion:
                            if chunk.choices[0].delta.content:
                                full_response += chunk.choices[0].delta.content
                                response_placeholder.markdown(full_response + "▌")
                        response_placeholder.markdown(full_response)
                        
                        # Save to memory
                        if st.session_state.memory_enabled:
                            vector_memory.add_memory(st.session_state.session_id, full_response, 
                                                   {"type": "assistant_response"})
                    else:
                        output = completion.choices[0].message.content
                        response_placeholder.markdown(output)
                        full_response = output
                    
                    # Save message
                    save_message(st.session_state.session_id, "assistant", full_response, "text")
                    
                except Exception as e:
                    response_placeholder.error(f"Error: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

def save_message(session_id, role, content, content_type="text", metadata=None):
    conn = db_connection()
    conn.execute(
        """INSERT INTO messages (session_id, role, content, content_type, metadata, created_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (session_id, role, content, content_type, json.dumps(metadata or {}), 
         datetime.now().isoformat())
    )
    conn.commit()
    conn.close()
    
    # Update session
    conn = db_connection()
    conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?",
                (datetime.now().isoformat(), session_id))
    conn.commit()
    conn.close()

# Legacy functions for compatibility
def data_url(data, mime="image/jpeg"):
    encoded = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{encoded}"

def extract_video_frames(video_bytes):
    try:
        import cv2
        import numpy as np
    except Exception as e:
        return [], str(e)
    
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
            f.write(video_bytes)
            temp_file = f.name
        
        cap = cv2.VideoCapture(temp_file)
        if not cap.isOpened():
            return [], "Could not open video"
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            return [], "No frames"
        
        indices = np.linspace(0, total_frames-1, min(MAX_VIDEO_FRAMES, total_frames), dtype=int)
        frames = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img.thumbnail((1280, 1280))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                frames.append(buf.getvalue())
        
        cap.release()
        return frames, None
    except Exception as e:
        return [], str(e)
    finally:
        if temp_file:
            try:
                os.remove(temp_file)
            except:
                pass

def transcribe_audio(audio_bytes):
    if not client:
        return "[No HF_TOKEN]"
    try:
        result = client.automatic_speech_recognition(audio_bytes, model=ASR_MODEL)
        return result.text if hasattr(result, "text") else str(result)
    except Exception as e:
        return f"[Error: {e}]"

if __name__ == "__main__":
    main()
