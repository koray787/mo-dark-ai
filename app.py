import os
import io
import re
import json
import uuid
import base64
import sqlite3
import mimetypes
import tempfile
import ipaddress
import socket
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse

import streamlit as st
import requests
import pandas as pd
from PIL import Image
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient


# ============================================================
# MO DARK AI
# ONE CHAT — AUTO UNDERSTANDS WHAT THE USER WANTS
# ============================================================

st.set_page_config(
    page_title="Mo Dark AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIG
# ============================================================

APP_NAME = "Mo Dark AI"

# Main coding / general model
TEXT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Vision model
VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

# Fallback vision model
VISION_FALLBACK = "Qwen/Qwen2.5-VL-72B-Instruct"

# Image generation
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Text -> video
VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B"

# Image -> video
IMAGE_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B"

# Speech recognition
ASR_MODEL = "openai/whisper-large-v3"

# Standard image size
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# Remote URL maximum download
MAX_REMOTE_BYTES = 25 * 1024 * 1024

# Video analysis
MAX_VIDEO_FRAMES = 6

DB_FILE = "mo_dark_memory.db"


# ============================================================
# TOKEN
# ============================================================

def get_hf_token():
    token = None

    try:
        token = st.secrets.get("HF_TOKEN")
    except Exception:
        pass

    if not token:
        token = os.getenv("HF_TOKEN")

    return token


HF_TOKEN = get_hf_token()


# ============================================================
# CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def get_client(token):
    if not token:
        return None

    return InferenceClient(
        api_key=token,
        provider="auto",
    )


client = get_client(HF_TOKEN)


# ============================================================
# DATABASE
# ============================================================

def db_connection():
    conn = sqlite3.connect(DB_FILE, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_database():
    conn = db_connection()

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


def create_session(title="محادثة جديدة"):
    session_id = str(uuid.uuid4())
    now = datetime.now().isoformat(timespec="seconds")

    conn = db_connection()

    conn.execute(
        """
        INSERT INTO sessions
        (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, title, now, now),
    )

    conn.commit()
    conn.close()

    return session_id


def update_session_title(session_id, title):
    title = title.strip()

    if not title:
        title = "محادثة جديدة"

    title = title[:80]

    conn = db_connection()

    conn.execute(
        """
        UPDATE sessions
        SET title = ?, updated_at = ?
        WHERE id = ?
        """,
        (
            title,
            datetime.now().isoformat(timespec="seconds"),
            session_id,
        ),
    )

    conn.commit()
    conn.close()


def touch_session(session_id):
    conn = db_connection()

    conn.execute(
        """
        UPDATE sessions
        SET updated_at = ?
        WHERE id = ?
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            session_id,
        ),
    )

    conn.commit()
    conn.close()


def save_message(session_id, role, content):
    conn = db_connection()

    conn.execute(
        """
        INSERT INTO messages
        (session_id, role, content, created_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            session_id,
            role,
            content,
            datetime.now().isoformat(timespec="seconds"),
        ),
    )

    conn.commit()
    conn.close()

    touch_session(session_id)


def load_sessions():
    conn = db_connection()

    rows = conn.execute(
        """
        SELECT *
        FROM sessions
        ORDER BY updated_at DESC
        """
    ).fetchall()

    conn.close()

    return rows


def load_messages(session_id):
    conn = db_connection()

    rows = conn.execute(
        """
        SELECT role, content, created_at
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
        """,
        (session_id,),
    ).fetchall()

    conn.close()

    return rows


init_database()


# ============================================================
# SESSION STATE
# ============================================================

if "session_id" not in st.session_state:
    st.session_state.session_id = create_session()

if "sidebar_visible" not in st.session_state:
    st.session_state.sidebar_visible = True

if "loaded_session" not in st.session_state:
    st.session_state.loaded_session = st.session_state.session_id


# ============================================================
# HTML HELPER
# ============================================================

def safe_html(markup):
    """
    Use Streamlit's native HTML renderer first.
    This prevents the raw <div>...</div> problem
    that appeared in the previous version.
    """

    try:
        st.html(markup)
    except Exception:
        st.markdown(markup, unsafe_allow_html=True)


# ============================================================
# CSS
# ============================================================

safe_html(
    """
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {
    --bg: #030509;
    --panel: #080c13;
    --panel2: #0d121b;
    --border: rgba(255,255,255,0.08);
    --cyan: #00eaff;
    --blue: #2878ff;
    --purple: #7c3cff;
    --text: #f4f7fb;
    --muted: #8993a4;
}

html, body, [class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 50% -10%, rgba(0,234,255,0.08), transparent 32%),
        radial-gradient(circle at 100% 100%, rgba(124,60,255,0.06), transparent 28%),
        #030509;
    color: var(--text);
}

/* Remove default top spacing */
.block-container {
    max-width: 1200px;
    padding-top: 1rem;
    padding-bottom: 7rem;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #05080e 0%,
            #070b12 55%,
            #05070b 100%
        );
    border-right: 1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

/* Buttons */
.stButton > button {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    background: rgba(255,255,255,0.035) !important;
    color: #f3f7ff !important;
    transition: 0.2s ease;
}

.stButton > button:hover {
    border-color: rgba(0,234,255,0.45) !important;
    background: rgba(0,234,255,0.07) !important;
    transform: translateY(-1px);
}

/* Chat */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageContent"] {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.055);
    border-radius: 18px;
    padding: 0.2rem 1rem;
}

/* Chat input */
[data-testid="stChatInput"] {
    border-radius: 18px !important;
}

[data-testid="stChatInput"] > div {
    background: rgba(9,13,21,0.96) !important;
    border: 1px solid rgba(0,234,255,0.18) !important;
    border-radius: 18px !important;
    box-shadow:
        0 0 30px rgba(0,234,255,0.04),
        0 15px 50px rgba(0,0,0,0.35);
}

/* Hide unnecessary Streamlit decoration */
#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: transparent;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.06) !important;
}

</style>
"""
)


# ============================================================
# TOP NAV
# ============================================================

col1, col2 = st.columns([1, 12])

with col1:
    if st.button(
        "☰",
        key="toggle_sidebar",
        help="إظهار / إخفاء القائمة الجانبية",
    ):
        st.session_state.sidebar_visible = not st.session_state.sidebar_visible
        st.rerun()

with col2:
    safe_html(
        """
        <div style="
            display:flex;
            align-items:center;
            gap:12px;
            padding:8px 0 18px 0;
        ">
            <div style="
                width:38px;
                height:38px;
                border-radius:12px;
                display:flex;
                align-items:center;
                justify-content:center;
                background:
                    linear-gradient(135deg,#00eaff,#2878ff,#7c3cff);
                box-shadow:
                    0 0 28px rgba(0,234,255,.25);
                font-size:20px;
            ">⚡</div>

            <div>
                <div style="
                    font-size:19px;
                    font-weight:800;
                    letter-spacing:-.5px;
                ">
                    Mo Dark AI
                </div>

                <div style="
                    font-size:11px;
                    color:#778296;
                    margin-top:2px;
                ">
                    MULTIMODAL INTELLIGENCE ENGINE
                </div>
            </div>
        </div>
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

if st.session_state.sidebar_visible:

    with st.sidebar:

        safe_html(
            """
            <div style="
                padding:8px 4px 18px 4px;
            ">
                <div style="
                    font-size:21px;
                    font-weight:800;
                    color:#fff;
                ">
                    Mo Dark
                </div>

                <div style="
                    font-size:11px;
                    color:#697589;
                    margin-top:3px;
                ">
                    AI WORKSPACE
                </div>
            </div>
            """
        )

        if st.button(
            "＋  محادثة جديدة",
            use_container_width=True,
            key="new_chat",
        ):
            new_id = create_session()
            st.session_state.session_id = new_id
            st.session_state.loaded_session = new_id
            st.rerun()

        st.markdown("---")

        st.markdown(
            "<div style='color:#8993a4;font-size:12px;"
            "font-weight:700;margin-bottom:10px;'>"
            "المحادثات السابقة"
            "</div>",
            unsafe_allow_html=True,
        )

        sessions = load_sessions()

        if not sessions:
            st.caption("لا توجد محادثات بعد.")

        for session in sessions:

            title = session["title"]

            if not title:
                title = "محادثة جديدة"

            is_current = (
                session["id"] == st.session_state.session_id
            )

            label = (
                "●  " if is_current else "○  "
            ) + title[:42]

            if st.button(
                label,
                key=f"session_{session['id']}",
                use_container_width=True,
            ):
                st.session_state.session_id = session["id"]
                st.session_state.loaded_session = session["id"]
                st.rerun()

        st.markdown("---")

        safe_html(
            """
            <div style="
                color:#5e697b;
                font-size:10px;
                line-height:1.7;
            ">
                MO DARK AI<br>
                ONE CHAT • MANY CAPABILITIES
            </div>
            """
        )


# ============================================================
# LOAD CURRENT MESSAGES
# ============================================================

messages = load_messages(
    st.session_state.session_id
)


# ============================================================
# WELCOME SCREEN
# ============================================================

if len(messages) == 0:

    safe_html(
        """
        <div style="
            margin:35px auto 35px auto;
            max-width:850px;
            text-align:center;
            padding:45px 25px;
        ">

            <div style="
                width:72px;
                height:72px;
                margin:auto;
                border-radius:22px;
                display:flex;
                align-items:center;
                justify-content:center;
                font-size:35px;
                background:
                    radial-gradient(
                        circle at 30% 20%,
                        #00eaff,
                        #2878ff 45%,
                        #32126f
                    );
                box-shadow:
                    0 0 50px rgba(0,234,255,.18);
            ">
                ⚡
            </div>

            <div style="
                margin-top:25px;
                font-size:42px;
                line-height:1.1;
                font-weight:800;
                letter-spacing:-2px;
            ">
                What can I build
                <span style="
                    color:#00eaff;
                ">for you?</span>
            </div>

            <div style="
                margin:18px auto 0 auto;
                max-width:650px;
                color:#7e899b;
                font-size:14px;
                line-height:1.8;
            ">
                اكتب طلبك فقط.
                Mo Dark AI يحدد تلقائياً إذا كان المطلوب
                برمجة، تحليل صورة، تحليل ملف، إنشاء صورة،
                إنشاء فيديو، قراءة رابط أو محادثة عادية.
            </div>

            <div style="
                display:flex;
                justify-content:center;
                flex-wrap:wrap;
                gap:8px;
                margin-top:25px;
            ">

                <span style="
                    padding:8px 12px;
                    border:1px solid rgba(0,234,255,.13);
                    border-radius:999px;
                    color:#8993a4;
                    font-size:11px;
                ">
                    💻 Code
                </span>

                <span style="
                    padding:8px 12px;
                    border:1px solid rgba(0,234,255,.13);
                    border-radius:999px;
                    color:#8993a4;
                    font-size:11px;
                ">
                    🖼️ Image
                </span>

                <span style="
                    padding:8px 12px;
                    border:1px solid rgba(0,234,255,.13);
                    border-radius:999px;
                    color:#8993a4;
                    font-size:11px;
                ">
                    🎬 Video
                </span>

                <span style="
                    padding:8px 12px;
                    border:1px solid rgba(0,234,255,.13);
                    border-radius:999px;
                    color:#8993a4;
                    font-size:11px;
                ">
                    📁 Files
                </span>

                <span style="
                    padding:8px 12px;
                    border:1px solid rgba(0,234,255,.13);
                    border-radius:999px;
                    color:#8993a4;
                    font-size:11px;
                ">
                    🌐 URLs
                </span>

            </div>
        </div>
        """
    )


# ============================================================
# FILE HELPERS
# ============================================================

IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
}

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".webm",
    ".m4v",
}

AUDIO_EXTENSIONS = {
    ".mp3",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
}

TEXT_EXTENSIONS = {
    ".txt",
    ".py",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".html",
    ".css",
    ".scss",
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".md",
    ".sql",
    ".sh",
    ".bat",
    ".cpp",
    ".c",
    ".h",
    ".java",
    ".kt",
    ".swift",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".r",
    ".dart",
    ".vue",
    ".svelte",
}


def get_extension(filename):
    return Path(filename).suffix.lower()


def file_kind(uploaded_file):

    name = uploaded_file.name.lower()
    ext = get_extension(name)

    mime = uploaded_file.type or ""

    if mime.startswith("image/") or ext in IMAGE_EXTENSIONS:
        return "image"

    if mime.startswith("video/") or ext in VIDEO_EXTENSIONS:
        return "video"

    if mime.startswith("audio/") or ext in AUDIO_EXTENSIONS:
        return "audio"

    if ext == ".pdf" or mime == "application/pdf":
        return "pdf"

    if ext == ".docx":
        return "docx"

    if ext in {".xlsx", ".xlsm", ".xls"}:
        return "excel"

    if ext == ".csv":
        return "csv"

    if ext in TEXT_EXTENSIONS:
        return "text"

    if ext == ".zip":
        return "zip"

    return "binary"


# ============================================================
# TEXT EXTRACTION
# ============================================================

def decode_text(data):
    for encoding in ("utf-8", "utf-8-sig", "cp1256", "latin-1"):
        try:
            return data.decode(encoding)
        except Exception:
            continue

    return data.decode("utf-8", errors="replace")


def extract_pdf_text(data):
    try:
        import fitz

        doc = fitz.open(
            stream=data,
            filetype="pdf",
        )

        parts = []

        for page in doc:
            text = page.get_text()

            if text:
                parts.append(text)

        doc.close()

        return "\n\n".join(parts)

    except Exception as e:
        return f"[PDF extraction failed: {e}]"


def extract_docx_text(data):

    try:
        from docx import Document

        doc = Document(
            io.BytesIO(data)
        )

        paragraphs = [
            p.text
            for p in doc.paragraphs
            if p.text.strip()
        ]

        return "\n".join(paragraphs)

    except Exception as e:
        return f"[DOCX extraction failed: {e}]"


def extract_excel_text(data):

    try:

        workbook = pd.ExcelFile(
            io.BytesIO(data)
        )

        output = []

        for sheet in workbook.sheet_names:

            df = pd.read_excel(
                workbook,
                sheet_name=sheet,
            )

            output.append(
                f"### SHEET: {sheet}\n"
            )

            output.append(
                df.head(200).to_csv(
                    index=False
                )
            )

        return "\n".join(output)

    except Exception as e:
        return f"[Excel extraction failed: {e}]"


def extract_csv_text(data):

    try:

        df = pd.read_csv(
            io.BytesIO(data)
        )

        return df.head(500).to_csv(
            index=False
        )

    except Exception as e:
        return f"[CSV extraction failed: {e}]"


def inspect_zip(data):

    try:

        with zipfile.ZipFile(
            io.BytesIO(data)
        ) as z:

            names = z.namelist()

            return (
                "ZIP ARCHIVE CONTENTS:\n"
                + "\n".join(
                    names[:1000]
                )
            )

    except Exception as e:
        return f"[ZIP inspection failed: {e}]"


# ============================================================
# IMAGE HELPERS
# ============================================================

def bytes_to_data_url(data, mime="image/jpeg"):

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return f"data:{mime};base64,{encoded}"


def image_to_png_bytes(image):

    output = io.BytesIO()

    image.save(
        output,
        format="PNG",
    )

    return output.getvalue()


# ============================================================
# VIDEO FRAME EXTRACTION
# ============================================================

def extract_video_frames(video_bytes):

    try:

        import cv2
        import numpy as np

    except Exception as e:
        return [], f"OpenCV unavailable: {e}"

    temp_path = None
    frames = []

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as temp:

            temp.write(video_bytes)
            temp_path = temp.name

        cap = cv2.VideoCapture(
            temp_path
        )

        if not cap.isOpened():
            return [], "Could not open video."

        total_frames = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:
            cap.release()
            return [], "Video contains no readable frames."

        count = min(
            MAX_VIDEO_FRAMES,
            total_frames,
        )

        indices = np.linspace(
            0,
            total_frames - 1,
            count,
            dtype=int,
        )

        for index in indices:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index),
            )

            success, frame = cap.read()

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(
                frame
            )

            # Resize large frames
            image.thumbnail(
                (1280, 1280)
            )

            frame_bytes = image_to_png_bytes(
                image
            )

            frames.append(
                frame_bytes
            )

        cap.release()

        return frames, None

    except Exception as e:
        return [], str(e)

    finally:

        if temp_path:

            try:
                os.remove(temp_path)
            except Exception:
                pass


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================

def transcribe_audio(audio_bytes):

    if not client:
        return "[HF_TOKEN is missing.]"

    try:

        result = client.automatic_speech_recognition(
            audio_bytes,
            model=ASR_MODEL,
        )

        if hasattr(result, "text"):
            return result.text

        return str(result)

    except Exception as e:
        return f"[Audio transcription failed: {e}]"


# ============================================================
# URL SECURITY
# ============================================================

def is_public_hostname(hostname):

    if not hostname:
        return False

    hostname = hostname.lower().strip()

    if hostname in {
        "localhost",
        "localhost.localdomain",
    }:
        return False

    try:

        addresses = socket.getaddrinfo(
            hostname,
            None,
        )

        for item in addresses:

            ip = item[4][0]

            parsed = ipaddress.ip_address(
                ip
            )

            if (
                parsed.is_private
                or parsed.is_loopback
                or parsed.is_link_local
                or parsed.is_reserved
                or parsed.is_multicast
            ):
                return False

        return True

    except Exception:
        return False


def safe_url(url):

    try:

        parsed = urlparse(url)

        if parsed.scheme not in {
            "http",
            "https",
        }:
            return False

        if not parsed.hostname:
            return False

        return is_public_hostname(
            parsed.hostname
        )

    except Exception:
        return False


def extract_urls(text):

    if not text:
        return []

    return re.findall(
        r"https?://[^\s<>\"]+",
        text,
        flags=re.IGNORECASE,
    )


# ============================================================
# URL INSPECTION
# ============================================================

def fetch_url(url):

    if not safe_url(url):
        return {
            "kind": "error",
            "text": "This URL is blocked or invalid.",
        }

    try:

        response = requests.get(
            url,
            timeout=20,
            stream=True,
            headers={
                "User-Agent":
                "Mozilla/5.0 Mo-Dark-AI"
            },
            allow_redirects=True,
        )

        response.raise_for_status()

        content_type = (
            response.headers
            .get("content-type", "")
            .lower()
        )

        content_length = response.headers.get(
            "content-length"
        )

        if content_length:

            try:

                if (
                    int(content_length)
                    > MAX_REMOTE_BYTES
                ):
                    return {
                        "kind": "error",
                        "text":
                        "Remote file is too large.",
                    }

            except Exception:
                pass

        chunks = []
        total = 0

        for chunk in response.iter_content(
            chunk_size=64 * 1024
        ):

            if not chunk:
                continue

            total += len(chunk)

            if total > MAX_REMOTE_BYTES:
                return {
                    "kind": "error",
                    "text":
                    "Remote content exceeded the safe size limit.",
                }

            chunks.append(chunk)

        data = b"".join(chunks)

        # Image URL
        if content_type.startswith(
            "image/"
        ):

            return {
                "kind": "image",
                "mime": content_type,
                "data": data,
                "text":
                f"Image URL: {url}",
            }

        # Video URL
        if content_type.startswith(
            "video/"
        ):

            return {
                "kind": "video",
                "mime": content_type,
                "data": data,
                "text":
                f"Video URL: {url}",
            }

        # Text / HTML
        if (
            "text/html" in content_type
            or "text/plain" in content_type
        ):

            encoding = (
                response.encoding
                or "utf-8"
            )

            text = data.decode(
                encoding,
                errors="replace",
            )

            if "text/html" in content_type:

                soup = BeautifulSoup(
                    text,
                    "html.parser",
                )

                for tag in soup(
                    [
                        "script",
                        "style",
                        "noscript",
                        "svg",
                    ]
                ):
                    tag.decompose()

                title = (
                    soup.title.get_text(
                        " ",
                        strip=True,
                    )
                    if soup.title
                    else ""
                )

                body = soup.get_text(
                    "\n",
                    strip=True,
                )

                body = body[:50000]

                return {
                    "kind": "html",
                    "title": title,
                    "text":
                    f"URL: {url}\n"
                    f"TITLE: {title}\n\n"
                    f"PAGE CONTENT:\n{body}",
                }

            return {
                "kind": "text",
                "text":
                text[:50000],
            }

        return {
            "kind": "binary",
            "mime": content_type,
            "data": data,
            "text":
            f"Downloaded URL: {url}\n"
            f"Content-Type: {content_type}\n"
            f"Size: {len(data)} bytes",
        }

    except Exception as e:

        return {
            "kind": "error",
            "text":
            f"Could not read URL:\n{e}",
        }


# ============================================================
# FILE -> AI CONTEXT
# ============================================================

def process_files(uploaded_files):

    text_context = []
    image_parts = []
    preview_items = []

    for uploaded_file in uploaded_files:

        try:
            data = uploaded_file.getvalue()

        except Exception:
            continue

        name = uploaded_file.name
        mime = (
            uploaded_file.type
            or mimetypes.guess_type(name)[0]
            or "application/octet-stream"
        )

        kind = file_kind(
            uploaded_file
        )

        preview_items.append(
            {
                "name": name,
                "kind": kind,
                "mime": mime,
                "size": len(data),
            }
        )

        # --------------------------
        # IMAGE
        # --------------------------

        if kind == "image":

            image_parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url":
                        bytes_to_data_url(
                            data,
                            mime,
                        )
                    },
                }
            )

            continue

        # --------------------------
        # VIDEO
        # --------------------------

        if kind == "video":

            frames, error = (
                extract_video_frames(
                    data
                )
            )

            text_context.append(
                f"""
VIDEO FILE:
{name}

MIME:
{mime}

SIZE:
{len(data)} bytes

The video was sampled into
{len(frames)} visual frames.
"""
            )

            for frame in frames:

                image_parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url":
                            bytes_to_data_url(
                                frame,
                                "image/png",
                            )
                        },
                    }
                )

            if error:
                text_context.append(
                    f"VIDEO PROCESSING NOTE: {error}"
                )

            continue

        # --------------------------
        # AUDIO
        # --------------------------

        if kind == "audio":

            transcript = (
                transcribe_audio(
                    data
                )
            )

            text_context.append(
                f"""
AUDIO FILE:
{name}

TRANSCRIPT:
{transcript}
"""
            )

            continue

        # --------------------------
        # PDF
        # --------------------------

        if kind == "pdf":

            text = extract_pdf_text(
                data
            )

            text_context.append(
                f"""
PDF FILE: {name}

CONTENT:
{text[:60000]}
"""
            )

            continue

        # --------------------------
        # DOCX
        # --------------------------

        if kind == "docx":

            text = extract_docx_text(
                data
            )

            text_context.append(
                f"""
WORD FILE: {name}

CONTENT:
{text[:60000]}
"""
            )

            continue

        # --------------------------
        # EXCEL
        # --------------------------

        if kind == "excel":

            text = extract_excel_text(
                data
            )

            text_context.append(
                f"""
EXCEL FILE: {name}

DATA:
{text[:60000]}
"""
            )

            continue

        # --------------------------
        # CSV
        # --------------------------

        if kind == "csv":

            text = extract_csv_text(
                data
            )

            text_context.append(
                f"""
CSV FILE: {name}

DATA:
{text[:60000]}
"""
            )

            continue

        # --------------------------
        # TEXT / CODE
        # --------------------------

        if kind == "text":

            text = decode_text(
                data
            )

            text_context.append(
                f"""
TEXT / CODE FILE:
{name}

CONTENT:
{text[:80000]}
"""
            )

            continue

        # --------------------------
        # ZIP
        # --------------------------

        if kind == "zip":

            text = inspect_zip(
                data
            )

            text_context.append(
                f"""
ZIP FILE:
{name}

{text}
"""
            )

            continue

        # --------------------------
        # UNKNOWN
        # --------------------------

        text_context.append(
            f"""
UNSUPPORTED / BINARY FILE:

Name: {name}
MIME: {mime}
Size: {len(data)} bytes

The file was accepted, but this
binary format is not directly parsed
by the application.
"""
        )

    return (
        "\n\n".join(text_context),
        image_parts,
        preview_items,
    )


# ============================================================
# SYSTEM PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are Mo Dark AI, a highly capable multimodal AI assistant.

You can help with:

- programming
- debugging
- software architecture
- Streamlit
- Python
- JavaScript
- HTML/CSS
- APIs
- databases
- automation
- data analysis
- documents
- images
- videos
- URLs
- normal conversation
- explanations
- advice
- writing
- research-style reasoning

IMPORTANT:

1. Follow the user's exact request.
2. Never change Streamlit into Flask/FastAPI/etc. unless the user asks.
3. When the user asks for code, provide complete usable code when appropriate.
4. For multi-file projects, keep imports, filenames and dependencies consistent.
5. Do not pretend that code was executed or tested if it was not.
6. If an image is supplied, inspect its visual content carefully.
7. If video frames are supplied, reason about the sequence and describe what can actually be inferred.
8. If documents are supplied, use their extracted content.
9. If a URL is supplied, use the retrieved page/media context.
10. If something cannot be verified, say so.
11. Do not invent facts.
12. For medical, legal or financial matters, clearly distinguish general information from professional advice.
13. Answer in the user's language when possible.
14. If the user asks for a complete project, do not intentionally omit important files.
15. Prefer practical solutions over vague explanations.

For coding requests use this workflow internally:

REQUEST
→ REQUIREMENTS
→ ARCHITECTURE
→ FILES
→ IMPLEMENTATION
→ ERROR REVIEW
→ DEPENDENCY REVIEW
→ FINAL ANSWER

Do not expose hidden chain-of-thought.
Provide concise reasoning summaries instead of private reasoning.
"""


# ============================================================
# AUTO INTENT DETECTION
# ============================================================

def normalize_text(text):

    return (
        (text or "")
        .strip()
        .lower()
    )


def wants_image_generation(text):

    t = normalize_text(text)

    phrases = [
        "انشئ صورة",
        "أنشئ صورة",
        "انشئلي صورة",
        "أنشئلي صورة",
        "سوي صورة",
        "سويلي صورة",
        "صمم صورة",
        "صمملی صورة",
        "ارسم صورة",
        "ولد صورة",
        "ولّد صورة",
        "توليد صورة",
        "generate image",
        "generate a picture",
        "create image",
        "create a picture",
        "make an image",
        "make a picture",
        "draw an image",
        "draw a picture",
    ]

    return any(
        phrase in t
        for phrase in phrases
    )


def wants_video_generation(text):

    t = normalize_text(text)

    phrases = [
        "انشئ فيديو",
        "أنشئ فيديو",
        "انشئلي فيديو",
        "أنشئلي فيديو",
        "سوي فيديو",
        "سويلي فيديو",
        "صمم فيديو",
        "صمملي فيديو",
        "ولد فيديو",
        "ولّد فيديو",
        "توليد فيديو",
        "سويها فيديو",
        "حولها لفيديو",
        "حول الصورة الى فيديو",
        "حول الصورة إلى فيديو",
        "image to video",
        "image-to-video",
        "generate video",
        "create video",
        "make a video",
        "text to video",
    ]

    return any(
        phrase in t
        for phrase in phrases
    )


def wants_image_edit(text):

    t = normalize_text(text)

    phrases = [
        "عدل الصورة",
        "عدّل الصورة",
        "تعديل الصورة",
        "غير الصورة",
        "غيّر الصورة",
        "حسن الصورة",
        "حسّن الصورة",
        "edit image",
        "edit this image",
        "modify image",
        "change this image",
        "enhance image",
    ]

    return any(
        phrase in t
        for phrase in phrases
    )


def has_image_file(files):

    return any(
        file_kind(f) == "image"
        for f in files
    )


def first_image_file(files):

    for file in files:
        if file_kind(file) == "image":
            return file

    return None


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def generate_image(prompt):

    if not client:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN to Streamlit Secrets."
        )

    image = client.text_to_image(
        prompt=prompt,
        model=IMAGE_MODEL,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    return image_to_png_bytes(
        image
    )


def generate_video(prompt):

    if not client:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN to Streamlit Secrets."
        )

    video = client.text_to_video(
        prompt=prompt,
        model=VIDEO_MODEL,
    )

    return bytes(video)


def image_to_video(image_bytes, prompt):

    if not client:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN to Streamlit Secrets."
        )

    video = client.image_to_video(
        image=image_bytes,
        prompt=prompt,
        model=IMAGE_VIDEO_MODEL,
    )

    return bytes(video)


def edit_image(image_bytes, prompt):

    if not client:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN to Streamlit Secrets."
        )

    image = client.image_to_image(
        image=image_bytes,
        prompt=prompt,
        model=IMAGE_MODEL,
    )

    return image_to_png_bytes(
        image
    )


# ============================================================
# CHAT MODEL
# ============================================================

def call_chat_model(
    user_prompt,
    history,
    text_context="",
    image_parts=None,
):

    if not client:
        raise RuntimeError(
            "HF_TOKEN is missing. Add HF_TOKEN to Streamlit Secrets."
        )

    image_parts = image_parts or []

    multimodal = bool(
        image_parts
    )

    model = (
        VISION_MODEL
        if multimodal
        else TEXT_MODEL
    )

    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    messages = [
        system_message
    ]

    # Keep useful recent history
    recent_history = history[-12:]

    for item in recent_history:

        messages.append(
            {
                "role": item["role"],
                "content": item["content"],
            }
        )

    context_parts = []

    if text_context:
        context_parts.append(
            """
ATTACHED / RETRIEVED CONTEXT:

Use this information to answer the user's request.
Do not mention internal processing unless relevant.

"""
            + text_context
        )

    if context_parts:
        context_text = "\n\n".join(
            context_parts
        )
    else:
        context_text = ""

    if multimodal:

        content = [
            {
                "type": "text",
                "text": (
                    user_prompt
                    + "\n\n"
                    + context_text
                    + "\n\n"
                    "Analyze the supplied visual content carefully."
                ),
            }
        ]

        content.extend(
            image_parts
        )

        messages.append(
            {
                "role": "user",
                "content": content,
            }
        )

    else:

        final_prompt = user_prompt

        if context_text:
            final_prompt += (
                "\n\n"
                + context_text
            )

        messages.append(
            {
                "role": "user",
                "content": final_prompt,
            }
        )

    try:

        response = client.chat_completion(
            model=model,
            messages=messages,
            max_tokens=8192,
            temperature=0.12,
        )

    except Exception as first_error:

        # Vision fallback
        if multimodal:

            try:

                response = client.chat_completion(
                    model=VISION_FALLBACK,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.12,
                )

            except Exception:
                raise first_error

        else:
            raise

    try:
        return response.choices[0].message.content

    except Exception:
        return str(response)


# ============================================================
# FILE PREVIEWS
# ============================================================

def render_file_preview(uploaded_file):

    kind = file_kind(
        uploaded_file
    )

    name = uploaded_file.name

    if kind == "image":

        try:
            uploaded_file.seek(0)

            st.image(
                uploaded_file,
                caption=name,
                width="stretch",
            )

        except Exception:
            st.caption(
                f"🖼️ {name}"
            )

    elif kind == "video":

        try:
            uploaded_file.seek(0)

            st.video(
                uploaded_file
            )

            st.caption(
                f"🎥 {name}"
            )

        except Exception:
            st.caption(
                f"🎥 {name}"
            )

    elif kind == "audio":

        try:
            uploaded_file.seek(0)

            st.audio(
                uploaded_file
            )

            st.caption(
                f"🎵 {name}"
            )

        except Exception:
            st.caption(
                f"🎵 {name}"
            )

    else:

        st.caption(
            f"📎 {name}"
        )


# ============================================================
# DISPLAY OLD MESSAGES
# ============================================================

for message in messages:

    role = message["role"]
    content = message["content"]

    if role not in {
        "user",
        "assistant",
    }:
        continue

    with st.chat_message(
        "user" if role == "user"
        else "assistant"
    ):

        st.markdown(
            content
        )


# ============================================================
# CHAT INPUT
# ============================================================

try:

    chat_value = st.chat_input(
        "اكتب أي شيء... أنشئ صورة، فيديو، برنامج، حلل ملف، اقرأ رابط، أو اسألني أي شيء",
        accept_file="multiple",
        file_type=None,
        max_upload_size=2048,
        key="main_chat",
    )

except TypeError:

    # Compatibility fallback for older Streamlit
    chat_value = st.chat_input(
        "اكتب أي شيء... أو أرفق ملف",
        accept_file="multiple",
        file_type=None,
        key="main_chat",
    )


# ============================================================
# HANDLE MESSAGE
# ============================================================

if chat_value:

    if isinstance(
        chat_value,
        str,
    ):

        user_prompt = chat_value
        uploaded_files = []

    else:

        user_prompt = (
            getattr(
                chat_value,
                "text",
                "",
            )
            or ""
        )

        try:

            uploaded_files = list(
                getattr(
                    chat_value,
                    "files",
                    [],
                )
                or []
            )

        except Exception:
            uploaded_files = []

    user_prompt = user_prompt.strip()

    # -----------------------------------------
    # If only files were uploaded
    # -----------------------------------------

    if (
        not user_prompt
        and uploaded_files
    ):

        user_prompt = (
            "حلل الملفات المرفقة بالتفصيل "
            "واشرح لي ماذا تحتوي وأهم الأشياء "
            "التي يمكن ملاحظتها."
        )

    if not user_prompt:
        st.stop()

    # -----------------------------------------
    # Session title
    # -----------------------------------------

    current_messages = load_messages(
        st.session_state.session_id
    )

    if len(current_messages) == 0:

        title = user_prompt.replace(
            "\n",
            " ",
        ).strip()

        if len(title) > 55:
            title = title[:55] + "..."

        update_session_title(
            st.session_state.session_id,
            title,
        )

    # -----------------------------------------
    # Save user message
    # -----------------------------------------

    file_names = [
        f.name
        for f in uploaded_files
    ]

    user_saved_content = user_prompt

    if file_names:

        user_saved_content += (
            "\n\n📎 الملفات: "
            + ", ".join(
                file_names
            )
        )

    save_message(
        st.session_state.session_id,
        "user",
        user_saved_content,
    )

    # -----------------------------------------
    # Show user message
    # -----------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_prompt
        )

        if uploaded_files:

            for uploaded_file in uploaded_files:

                with st.expander(
                    f"📎 {uploaded_file.name}",
                    expanded=False,
                ):

                    render_file_preview(
                        uploaded_file
                    )

    # -----------------------------------------
    # Process attachments
    # -----------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Mo Dark AI يعمل..."
        ):

            try:

                text_context = ""
                image_parts = []
                preview_items = []

                if uploaded_files:

                    (
                        text_context,
                        image_parts,
                        preview_items,
                    ) = process_files(
                        uploaded_files
                    )

                # ---------------------------------
                # URL processing
                # ---------------------------------

                urls = extract_urls(
                    user_prompt
                )

                url_context = []
                url_images = []

                for url in urls[:3]:

                    result = fetch_url(
                        url
                    )

                    kind = result.get(
                        "kind"
                    )

                    if kind == "image":

                        url_images.append(
                            {
                                "type":
                                "image_url",
                                "image_url": {
                                    "url":
                                    bytes_to_data_url(
                                        result["data"],
                                        result.get(
                                            "mime",
                                            "image/jpeg",
                                        ),
                                    )
                                },
                            }
                        )

                        url_context.append(
                            f"IMAGE URL:\n{url}"
                        )

                    elif kind == "video":

                        frames, error = (
                            extract_video_frames(
                                result["data"]
                            )
                        )

                        url_context.append(
                            f"VIDEO URL:\n{url}\n"
                            f"Sampled frames: {len(frames)}"
                        )

                        for frame in frames:

                            url_images.append(
                                {
                                    "type":
                                    "image_url",
                                    "image_url": {
                                        "url":
                                        bytes_to_data_url(
                                            frame,
                                            "image/png",
                                        )
                                    },
                                }
                            )

                        if error:
                            url_context.append(
                                f"Video note: {error}"
                            )

                    else:

                        url_context.append(
                            result.get(
                                "text",
                                "",
                            )
                        )

                if url_context:

                    text_context += (
                        "\n\n"
                        + "\n\n".join(
                            url_context
                        )
                    )

                image_parts.extend(
                    url_images
                )

                # ---------------------------------
                # Auto route
                # ---------------------------------

                has_img = has_image_file(
                    uploaded_files
                )

                img_file = first_image_file(
                    uploaded_files
                )

                # VIDEO GENERATION
                if wants_video_generation(
                    user_prompt
                ):

                    if (
                        has_img
                        and (
                            "حول" in normalize_text(
                                user_prompt
                            )
                            or
                            "تحويل" in normalize_text(
                                user_prompt
                            )
                            or
                            "image" in normalize_text(
                                user_prompt
                            )
                        )
                    ):

                        image_bytes = (
                            img_file.getvalue()
                        )

                        video_bytes = (
                            image_to_video(
                                image_bytes,
                                user_prompt,
                            )
                        )

                        st.success(
                            "🎬 تم إنشاء الفيديو"
                        )

                        st.video(
                            video_bytes
                        )

                        st.download_button(
                            "⬇️ تنزيل الفيديو",
                            data=video_bytes,
                            file_name="mo_dark_video.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                        answer = (
                            "🎬 تم إنشاء الفيديو "
                            "من الصورة والطلب."
                        )

                    else:

                        video_bytes = (
                            generate_video(
                                user_prompt
                            )
                        )

                        st.success(
                            "🎬 تم إنشاء الفيديو"
                        )

                        st.video(
                            video_bytes
                        )

                        st.download_button(
                            "⬇️ تنزيل الفيديو",
                            data=video_bytes,
                            file_name="mo_dark_video.mp4",
                            mime="video/mp4",
                            use_container_width=True,
                        )

                        answer = (
                            "🎬 تم إنشاء الفيديو "
                            "حسب طلبك."
                        )

                # IMAGE EDIT
                elif (
                    has_img
                    and wants_image_edit(
                        user_prompt
                    )
                ):

                    image_bytes = (
                        img_file.getvalue()
                    )

                    output = edit_image(
                        image_bytes,
                        user_prompt,
                    )

                    st.success(
                        "🖼️ تم تعديل الصورة"
                    )

                    st.image(
                        output,
                        width="stretch",
                    )

                    st.download_button(
                        "⬇️ تنزيل الصورة",
                        data=output,
                        file_name="mo_dark_edited.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                    answer = (
                        "🖼️ تم تعديل الصورة "
                        "حسب طلبك."
                    )

                # IMAGE GENERATION
                elif wants_image_generation(
                    user_prompt
                ):

                    image_bytes = (
                        generate_image(
                            user_prompt
                        )
                    )

                    st.success(
                        "🖼️ تم إنشاء الصورة"
                    )

                    st.image(
                        image_bytes,
                        width="stretch",
                    )

                    st.download_button(
                        "⬇️ تنزيل الصورة",
                        data=image_bytes,
                        file_name="mo_dark_image.png",
                        mime="image/png",
                        use_container_width=True,
                    )

                    answer = (
                        "🖼️ تم إنشاء الصورة "
                        "بالمقاس القياسي 1024×1024."
                    )

                # NORMAL AI / VISION / FILE / URL
                else:

                    # Build history
                    history_rows = (
                        load_messages(
                            st.session_state.session_id
                        )
                    )

                    history = []

                    for row in history_rows[-12:]:

                        if row["role"] in {
                            "user",
                            "assistant",
                        }:

                            history.append(
                                {
                                    "role":
                                    row["role"],
                                    "content":
                                    row["content"],
                                }
                            )

                    # Remove current saved user message
                    # from history because it is passed
                    # separately below.
                    if history and history[-1][
                        "role"
                    ] == "user":

                        history = history[:-1]

                    answer = call_chat_model(
                        user_prompt=user_prompt,
                        history=history,
                        text_context=text_context,
                        image_parts=image_parts,
                    )

                    st.markdown(
                        answer
                    )

                # ---------------------------------
                # Save assistant response
                # ---------------------------------

                save_message(
                    st.session_state.session_id,
                    "assistant",
                    answer,
                )

            except Exception as e:

                error_message = (
                    "حدث خطأ أثناء تنفيذ الطلب.\n\n"
                    f"```text\n{str(e)}\n```\n\n"
                    "إذا كان الخطأ متعلقاً بالنموذج أو "
                    "الخدمة، قد يكون النموذج غير متاح "
                    "حالياً عبر مزود Hugging Face."
                )

                st.error(
                    error_message
                )

                save_message(
                    st.session_state.session_id,
                    "assistant",
                    error_message,
                )


# ============================================================
# FOOTER
# ============================================================

safe_html(
    """
<div style="
    position:fixed;
    bottom:8px;
    left:0;
    right:0;
    text-align:center;
    pointer-events:none;
    z-index:1;
">
    <span style="
        color:#394354;
        font-size:9px;
        letter-spacing:1px;
    ">
        MO DARK AI • MULTIMODAL INTELLIGENCE
    </span>
</div>
"""
)
