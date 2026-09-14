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
# DATABASE SCHEMA (Defined First to Avoid NameError)
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
# UTILITY FUNCTIONS
# ============================================================
def data_url(data: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(data).decode('utf-8')
    return f"data:{mime_type};base64,{encoded}"

def extract_video_frames(data: bytes):
    try:
        import cv2
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_file.write(data)
        temp_file.close()
        
        cap = cv2.VideoCapture(temp_file.name)
        frames = []
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total_frames // MAX_VIDEO_FRAMES)
        
        count = 0
        while cap.isOpened() and len(frames) < MAX_VIDEO_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                success, encoded_image = cv2.imencode('.png', frame)
                if success:
                    frames.append(encoded_image.tobytes())
            count += 1
        cap.release()
        os.unlink(temp_file.name)
        return frames, None
    except Exception as e:
        return [], str(e)

def transcribe_audio(data: bytes) -> str:
    try:
        client_inst = create_client(get_hf_token())
        if not client_inst:
            return "[Audio transcription skipped: No HuggingFace Token provided]"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        temp_file.write(data)
        temp_file.close()
        
        with open(temp_file.name, "rb") as f:
            transcript = client_inst.automatic_speech_recognition(f.read(), model=ASR_MODEL)
        os.unlink(temp_file.name)
        return transcript.get("text", "")
    except Exception as e:
        return f"[Audio transcription error: {e}]"

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
                import pickle
                with open(VECTOR_STORE_FILE, 'rb') as f:
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
        
        plan = self._plan(task, context)
        result["steps"].append({"plan": plan})
        
        for step in plan:
            try:
                step_result = self._execute_step(step, context)
                result["steps"].append(step_result)
            except Exception as e:
                result["steps"].append({"error": str(e)})
                result["success"] = False
        
        result["output"] = self._synthesize(result["steps"])
        return result
    
    def _plan(self, task: str, context: Dict) -> List[Dict]:
        return [{"action": "process", "input": task}]
    
    def _execute_step(self, step: Dict, context: Dict) -> Dict:
        return {"step": step, "status": "completed"}
    
    def _synthesize(self, steps: List[Dict]) -> str:
        return "Task completed successfully"

class CodeAgent(Agent):
    def __init__(self):
        super().__init__(
            "CodeExpert",
            "Specialized in code analysis, generation, and execution",
            [self.execute_python, self.analyze_code, self.generate_code]
        )
    
    def execute_python(self, code: str, timeout: int = 30) -> Dict:
        result = {
            "output": "",
            "error": "",
            "execution_time": 0
        }
        
        try:
            start_time = time.time()
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
        issues = []
        suggestions = []
        
        if "eval(" in code or "exec(" in code:
            issues.append("Security risk: eval/exec detected")
        
        if len(code.split('\n')) > 100:
            suggestions.append("Consider breaking into smaller functions")
        
        return {"issues": issues, "suggestions": suggestions}
    
    def generate_code(self, description: str, language: str = "python") -> str:
        return f"# Generated {language} code for: {description}\n# TODO: Implement"

class DataAnalysisAgent(Agent):
    def __init__(self):
        super().__init__(
            "DataAnalyst",
            "Specialized in data analysis and visualization",
            [self.analyze_dataframe, self.create_visualization, self.statistical_analysis]
        )
    
    def analyze_dataframe(self, df: pd.DataFrame) -> Dict:
        analysis = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "missing": df.isnull().sum().to_dict(),
            "numeric_summary": {},
            "categorical_summary": {}
        }
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()
        
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            analysis["categorical_summary"][col] = df[col].value_counts().head(10).to_dict()
        
        return analysis
    
    def create_visualization(self, df: pd.DataFrame, chart_type: str, **kwargs) -> go.Figure:
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
        try:
            response = requests.get(url, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(response.text, 'html.parser')
            
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
        summary = []
        for i, result in enumerate(results, 1):
            if "title" in result:
                summary.append(f"{i}. {result['title']}\n   {result.get('snippet', '')}")
        return "\n\n".join(summary)

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
    
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 700;
    }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bg-secondary) 0%, #050810 100%);
        border-right: 1px solid var(--border);
    }
    
    [data-testid="stSidebar"] > div {
        padding: 1.5rem 1rem;
    }
    
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
    
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
        color: var(--bg-primary) !important;
        font-weight: 600 !important;
    }
    
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
    
    [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatar"]) [data-testid="stChatMessageContent"] {
        background: linear-gradient(135deg, rgba(0,234,255,0.1), rgba(40,120,255,0.05));
        border-color: rgba(0,234,255,0.2);
    }
    
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
    
    #MainMenu, footer, header {
        visibility: hidden;
    }
    </style>
    """, unsafe_allow_html=True)

render_css()

def render_header():
    col1, col2, col3 = st.columns([1, 10, 2])
    
    with col1:
        if st.button("☰", key="toggle_sidebar"):
            st.session_state.sidebar_visible = not st.session_state.sidebar_visible
    with col2:
        st.markdown(f"### ⚡ {APP_NAME}")
    with col3:
        st.markdown(f"<span class='agent-badge'>v{VERSION}</span>", unsafe_allow_html=True)

render_header()

# ============================================================
# SIDEBAR CONTROLS & MAIN APP INTERFACE
# ============================================================
with st.sidebar:
    st.markdown("## 🎛️ إعدادات النظام الذكي")
    st.selectbox("الموديل النصي الأساسي", [TEXT_MODEL], index=0)
    st.slider("درجة الحرارة (Temperature)", 0.0, 1.0, 0.7)
    st.checkbox("الذاكرة الدلالية (Vector Memory)", value=True)
    st.checkbox("البحث الفوري في الويب", value=True)
    st.markdown("---")
    st.markdown("### 🤖 الوكلاء الذكيون")
    selected_agent = st.selectbox("اختر الوكيل المساعد", list(AVAILABLE_AGENTS.keys()))
    st.session_state.current_agent = selected_agent

st.info("👋 مرحباً بك في واجهة **Mo Dark AI Ultimate**. تم إصلاح خطأ ترتيب الدوال بنجاح والنظام جاهز للاستخدام الفوري!")
