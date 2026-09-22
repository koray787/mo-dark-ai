import os
import io
import re
import json
import uuid
import time
import base64
import sqlite3
import mimetypes
import tempfile
import subprocess
import zipfile
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageEnhance, ImageFilter
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient

# Optional packages are handled gracefully.
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_OK = True
except Exception:
    SKLEARN_OK = False

try:
    import fitz
    FITZ_OK = True
except Exception:
    FITZ_OK = False

try:
    import docx
    DOCX_OK = True
except Exception:
    DOCX_OK = False

try:
    import plotly.express as px
    PLOTLY_OK = True
except Exception:
    PLOTLY_OK = False

# Advanced packages for the enhanced version
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_OK = True
except Exception:
    SENTENCE_TRANSFORMERS_OK = False

try:
    import chromadb
    CHROMADB_OK = True
except Exception:
    CHROMADB_OK = False

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_OK = True
except Exception:
    LANGCHAIN_OK = False

try:
    import streamlit.components.v1 as components
    COMPONENTS_OK = True
except Exception:
    COMPONENTS_OK = False


# ============================================================
# MO DARK AI — OMNIBRAIN EDITION
# Combines:
# - Multi-model text / vision / image selection
# - Web search + URL reader
# - Files + documents + spreadsheets
# - Persistent SQLite chat history
# - Advanced vector memory with ChromaDB
# - Agents / automatic routing
# - Data analysis
# - Image understanding
# - Image generation
# - Code generation / project packaging
# - Premium 3D-like UI with animations
#
# IMPORTANT:
# This version intentionally does NOT execute arbitrary shell commands
# or unrestricted Python from the public web app. That would expose the
# Streamlit server. The Code Lab generates/reviews code instead.
# ============================================================

st.set_page_config(
    page_title="Mo Dark AI — OmniBrain",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "Mo Dark AI"
VERSION = "5.0 OmniBrain"

# Streamlit Cloud's application directory can be read-only.
# Store the SQLite database in a writable runtime directory.
DATA_DIR = Path("/tmp/mo_dark_ai")
DATA_DIR.mkdir(parents=True, exist_ok=True)
DB_FILE = str(DATA_DIR / "mo_dark_omnibrain.db")
MAX_HISTORY = 40
MAX_SEARCH_RESULTS = 6
WEB_TIMEOUT = 15
MAX_TEXT_CHARS = 120_000
MAX_IMAGE_SIDE = 1536

# Enhanced model list with more options
TEXT_MODELS = {
    "Qwen Coder 32B": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "Qwen 72B": "Qwen/Qwen2.5-72B-Instruct",
    "Llama 3.3 70B": "meta-llama/Llama-3.3-70B-Instruct",
    "Mixtral 8x7B": "mistralai/Mixtral-8x7B-Instruct-v0.1",
    "DeepSeek R1": "deepseek-ai/DeepSeek-R1",
    "Phi-4": "microsoft/phi-4",
    "Mistral Large v3": "mistralai/Mistral-Large-Instruct-2407",
    "Gemma 2 27B": "google/gemma-2-27b-it",
}

VISION_MODELS = {
    "Qwen VL 7B": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen VL 72B": "Qwen/Qwen2.5-VL-72B-Instruct",
    "Llava 1.6 34B": "llava-hf/llava-v1.6-34b-hf",
    "Fuyu 8B": "adept/fuyu-8b",
}

IMAGE_MODELS = {
    "FLUX Schnell": "black-forest-labs/FLUX.1-schnell",
    "FLUX Dev": "black-forest-labs/FLUX.1-dev",
    "SDXL Turbo": "stabilityai/sdxl-turbo",
    "SD 3 Medium": "stabilityai/stable-diffusion-3-medium",
}

# Enhanced language support
LANGUAGE_NAMES = {
    "ar": "العربية",
    "en": "English",
    "zh": "中文",
    "es": "Español",
    "fr": "Français",
    "de": "Deutsch",
    "ja": "日本語",
}


# ============================================================
# DATABASE
# ============================================================

def db():
    """Open a SQLite connection that is safe for Streamlit Cloud runtime."""
    conn = sqlite3.connect(
        DB_FILE,
        timeout=30,
        check_same_thread=False,
    )
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def init_db():
    conn = db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            content_type TEXT DEFAULT 'text',
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS files (
            id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            file_type TEXT NOT NULL,
            extracted_text TEXT,
            created_at TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT PRIMARY KEY,
            preferences TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def create_session(title="محادثة جديدة"):
    sid = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")
    conn = db()
    conn.execute(
        """
        INSERT INTO sessions (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (sid, title, now, now),
    )
    conn.commit()
    conn.close()
    return sid


def ensure_state():
    init_db()
    if "session_id" not in st.session_state:
        st.session_state.session_id = create_session()
    if "language" not in st.session_state:
        st.session_state.language = "ar"
    if "text_model_name" not in st.session_state:
        st.session_state.text_model_name = "Qwen Coder 32B"
    if "vision_model_name" not in st.session_state:
        st.session_state.vision_model_name = "Qwen VL 7B"
    if "image_model_name" not in st.session_state:
        st.session_state.image_model_name = "FLUX Schnell"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 4096
    if "memory_enabled" not in st.session_state:
        st.session_state.memory_enabled = True
    if "web_enabled" not in st.session_state:
        st.session_state.web_enabled = True
    if "auto_agent" not in st.session_state:
        st.session_state.auto_agent = True
    if "last_agent" not in st.session_state:
        st.session_state.last_agent = "General"
    if "omnimode_enabled" not in st.session_state:
        st.session_state.omnimode_enabled = False
    if "vector_memory_enabled" not in st.session_state:
        st.session_state.vector_memory_enabled = True
    if "ui_theme" not in st.session_state:
        st.session_state.ui_theme = "cyberpunk"


ensure_state()


# ============================================================
# TOKEN / CLIENT
# ============================================================

def get_token():
    try:
        token = st.secrets.get("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass
    return os.getenv("HF_TOKEN")


HF_TOKEN = get_token()


@st.cache_resource(show_spinner=False)
def make_client(token):
    if not token:
        return None
    try:
        return InferenceClient(api_key=token, provider="auto")
    except Exception:
        return None


client = make_client(HF_TOKEN)


# =================================================
