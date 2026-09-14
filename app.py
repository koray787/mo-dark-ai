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
# ============================================================

st.set_page_config(
    page_title="Mo Dark AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# SETTINGS
# ============================================================

APP_NAME = "Mo Dark AI"

# General / coding model
TEXT_MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# Vision
VISION_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
VISION_FALLBACK = "Qwen/Qwen2.5-VL-72B-Instruct"

# Image generation
IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"

# Text -> Video
VIDEO_MODEL = "Wan-AI/Wan2.2-TI2V-5B"

# Image -> Video
IMAGE_VIDEO_MODEL = "Wan-AI/Wan2.2-I2V-A14B"

# Audio
ASR_MODEL = "openai/whisper-large-v3"

# Standard image output
IMAGE_WIDTH = 1024
IMAGE_HEIGHT = 1024

# Remote URL limit
MAX_REMOTE_BYTES = 25 * 1024 * 1024

# Number of sampled video frames
MAX_VIDEO_FRAMES = 6

# Local database
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
# HUGGING FACE CLIENT
# ============================================================

@st.cache_resource(show_spinner=False)
def create_client(token):
    if not token:
        return None

    return InferenceClient(
        api_key=token,
        provider="auto",
    )


client = create_client(HF_TOKEN)


# ============================================================
# DATABASE
# ============================================================

def db_connection():
    conn = sqlite3.connect(
        DB_FILE,
        check_same_thread=False,
    )

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

    now = datetime.now().isoformat(
        timespec="seconds"
    )

    conn = db_connection()

    conn.execute(
        """
        INSERT INTO sessions
        (id, title, created_at, updated_at)
        VALUES (?, ?, ?, ?)
        """,
        (
            session_id,
            title,
            now,
            now,
        ),
    )

    conn.commit()
    conn.close()

    return session_id


def update_session_title(
    session_id,
    title,
):

    title = (
        title
        .replace("\n", " ")
        .strip()
    )

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
            datetime.now().isoformat(
                timespec="seconds"
            ),
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
            datetime.now().isoformat(
                timespec="seconds"
            ),
            session_id,
        ),
    )

    conn.commit()
    conn.close()


def save_message(
    session_id,
    role,
    content,
):

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
            datetime.now().isoformat(
                timespec="seconds"
            ),
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


# ============================================================
# HTML
# ============================================================

def render_html(markup):

    try:
        st.html(markup)
    except Exception:
        st.markdown(
            markup,
            unsafe_allow_html=True,
        )


# ============================================================
# CSS
# ============================================================

render_html(
    """
<style>

@import url(
'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
);

:root {
    --bg: #030509;
    --panel: #080c13;
    --cyan: #00eaff;
    --blue: #2878ff;
    --purple: #7c3cff;
    --text: #f5f7fb;
    --muted: #7d8798;
}

html,
body,
[class*="css"] {
    font-family: Inter, sans-serif;
}

.stApp {
    background:
        radial-gradient(
            circle at 50% -10%,
            rgba(0,234,255,0.10),
            transparent 34%
        ),
        radial-gradient(
            circle at 100% 100%,
            rgba(124,60,255,0.07),
            transparent 30%
        ),
        #030509;
    color: var(--text);
}

.block-container {
    max-width: 1180px;
    padding-top: 0.7rem;
    padding-bottom: 7rem;
}

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #05080e,
            #070b12 55%,
            #05070b
        );
    border-right:
        1px solid rgba(255,255,255,0.07);
}

section[data-testid="stSidebar"] > div {
    padding-top: 1rem;
}

.stButton > button {
    border-radius: 12px !important;
    border:
        1px solid rgba(255,255,255,0.08) !important;
    background:
        rgba(255,255,255,0.035) !important;
    color: #f4f7fb !important;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    border-color:
        rgba(0,234,255,0.45) !important;
    background:
        rgba(0,234,255,0.07) !important;
}

[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
}

[data-testid="stChatMessageContent"] {
    background:
        rgba(255,255,255,0.025);
    border:
        1px solid rgba(255,255,255,0.055);
    border-radius: 18px;
}

[data-testid="stChatInput"] {
    border-radius: 18px !important;
}

[data-testid="stChatInput"] > div {
    background:
        rgba(7,11,18,0.98) !important;
    border:
        1px solid rgba(0,234,255,0.20) !important;
    border-radius: 18px !important;
    box-shadow:
        0 0 35px rgba(0,234,255,0.04),
        0 15px 60px rgba(0,0,0,0.40);
}

[data-testid="stFileUploader"] {
    background: transparent;
}

hr {
    border-color:
        rgba(255,255,255,0.06) !important;
}

</style>
"""
)


# ============================================================
# TOP BAR
# ============================================================

top1, top2 = st.columns(
    [1, 12]
)

with top1:

    if st.button(
        "☰",
        key="sidebar_toggle",
        help="إظهار / إخفاء القائمة",
    ):

        st.session_state.sidebar_visible = (
            not st.session_state.sidebar_visible
        )

        st.rerun()


with top2:

    render_html(
        """
<div style="
    display:flex;
    align-items:center;
    gap:12px;
    padding:6px 0 15px 0;
">

    <div style="
        width:40px;
        height:40px;
        border-radius:13px;
        display:flex;
        align-items:center;
        justify-content:center;
        background:
            linear-gradient(
                135deg,
                #00eaff,
                #2878ff,
                #7c3cff
            );
        box-shadow:
            0 0 30px rgba(0,234,255,.25);
        font-size:21px;
    ">
        ⚡
    </div>

    <div>

        <div style="
            font-size:19px;
            font-weight:800;
            color:#f6f8fb;
        ">
            Mo Dark AI
        </div>

        <div style="
            font-size:10px;
            color:#6f7a8c;
            letter-spacing:1px;
            margin-top:2px;
        ">
            MULTIMODAL INTELLIGENCE
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

        render_html(
            """
<div style="
    padding:6px 4px 17px 4px;
">

    <div style="
        font-size:22px;
        font-weight:800;
        color:#fff;
    ">
        Mo Dark
    </div>

    <div style="
        font-size:10px;
        color:#667185;
        letter-spacing:1px;
        margin-top:4px;
    ">
        AI WORKSPACE
    </div>

</div>
"""
        )

        if st.button(
            "＋  محادثة جديدة",
            use_container_width=True,
            key="create_new_chat",
        ):

            new_session = create_session()

            st.session_state.session_id = (
                new_session
            )

            st.rerun()

        st.markdown("---")

        st.markdown(
            """
<div style="
    color:#7e899b;
    font-size:11px;
    font-weight:700;
    margin-bottom:10px;
">
المحادثات السابقة
</div>
""",
            unsafe_allow_html=True,
        )

        sessions = load_sessions()

        if not sessions:

            st.caption(
                "لا توجد محادثات."
            )

        for session in sessions:

            session_id = session["id"]
            title = session["title"]

            if not title:
                title = "محادثة جديدة"

            current = (
                session_id
                == st.session_state.session_id
            )

            icon = "●" if current else "○"

            label = (
                f"{icon}  "
                f"{title[:42]}"
            )

            if st.button(
                label,
                key=f"open_{session_id}",
                use_container_width=True,
            ):

                st.session_state.session_id = (
                    session_id
                )

                st.rerun()

        st.markdown("---")

        render_html(
            """
<div style="
    color:#4e596b;
    font-size:9px;
    line-height:1.8;
">
    MO DARK AI<br>
    ONE CHAT • EVERYTHING
</div>
"""
        )


# ============================================================
# FILE TYPES
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


def extension_of(filename):
    return Path(filename).suffix.lower()


def detect_file_type(uploaded_file):

    filename = uploaded_file.name.lower()

    extension = extension_of(
        filename
    )

    mime = (
        uploaded_file.type
        or ""
    ).lower()

    if (
        mime.startswith("image/")
        or extension in IMAGE_EXTENSIONS
    ):
        return "image"

    if (
        mime.startswith("video/")
        or extension in VIDEO_EXTENSIONS
    ):
        return "video"

    if (
        mime.startswith("audio/")
        or extension in AUDIO_EXTENSIONS
    ):
        return "audio"

    if (
        extension == ".pdf"
        or mime == "application/pdf"
    ):
        return "pdf"

    if extension == ".docx":
        return "docx"

    if extension in {
        ".xlsx",
        ".xlsm",
        ".xls",
    }:
        return "excel"

    if extension == ".csv":
        return "csv"

    if extension in TEXT_EXTENSIONS:
        return "text"

    if extension == ".zip":
        return "zip"

    return "binary"


# ============================================================
# TEXT / DOCUMENT EXTRACTION
# ============================================================

def decode_bytes(data):

    for encoding in (
        "utf-8",
        "utf-8-sig",
        "cp1256",
        "latin-1",
    ):

        try:
            return data.decode(
                encoding
            )
        except Exception:
            pass

    return data.decode(
        "utf-8",
        errors="replace",
    )


def extract_pdf(data):

    try:

        import fitz

        document = fitz.open(
            stream=data,
            filetype="pdf",
        )

        pages = []

        for page in document:

            text = page.get_text()

            if text:
                pages.append(text)

        document.close()

        return "\n\n".join(
            pages
        )

    except Exception as e:

        return (
            "[PDF extraction error] "
            + str(e)
        )


def extract_docx(data):

    try:

        from docx import Document

        document = Document(
            io.BytesIO(data)
        )

        result = []

        for paragraph in document.paragraphs:

            text = paragraph.text.strip()

            if text:
                result.append(text)

        return "\n".join(result)

    except Exception as e:

        return (
            "[DOCX extraction error] "
            + str(e)
        )


def extract_excel(data):

    try:

        workbook = pd.ExcelFile(
            io.BytesIO(data)
        )

        result = []

        for sheet in workbook.sheet_names:

            dataframe = pd.read_excel(
                workbook,
                sheet_name=sheet,
            )

            result.append(
                f"### SHEET: {sheet}"
            )

            result.append(
                dataframe.head(
                    300
                ).to_csv(
                    index=False
                )
            )

        return "\n".join(result)

    except Exception as e:

        return (
            "[Excel extraction error] "
            + str(e)
        )


def extract_csv(data):

    try:

        dataframe = pd.read_csv(
            io.BytesIO(data)
        )

        return dataframe.head(
            1000
        ).to_csv(
            index=False
        )

    except Exception as e:

        return (
            "[CSV extraction error] "
            + str(e)
        )


def inspect_zip(data):

    try:

        with zipfile.ZipFile(
            io.BytesIO(data)
        ) as archive:

            names = archive.namelist()

            return (
                "ZIP CONTENTS:\n"
                + "\n".join(
                    names[:1500]
                )
            )

    except Exception as e:

        return (
            "[ZIP inspection error] "
            + str(e)
        )


# ============================================================
# IMAGE
# ============================================================

def data_url(
    data,
    mime="image/jpeg",
):

    encoded = base64.b64encode(
        data
    ).decode("utf-8")

    return (
        f"data:{mime};base64,"
        f"{encoded}"
    )


def image_to_bytes(image):

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="PNG",
    )

    return buffer.getvalue()


# ============================================================
# VIDEO
# ============================================================

def extract_video_frames(
    video_bytes,
):

    try:

        import cv2
        import numpy as np

    except Exception as e:

        return [], str(e)

    temporary_file = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4",
        ) as file:

            file.write(
                video_bytes
            )

            temporary_file = (
                file.name
            )

        capture = cv2.VideoCapture(
            temporary_file
        )

        if not capture.isOpened():

            return [], (
                "Could not open video."
            )

        total_frames = int(
            capture.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        )

        if total_frames <= 0:

            capture.release()

            return [], (
                "No readable frames."
            )

        number = min(
            MAX_VIDEO_FRAMES,
            total_frames,
        )

        positions = np.linspace(
            0,
            total_frames - 1,
            number,
            dtype=int,
        )

        frames = []

        for position in positions:

            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(position),
            )

            success, frame = (
                capture.read()
            )

            if not success:
                continue

            frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

            image = Image.fromarray(
                frame
            )

            image.thumbnail(
                (1280, 1280)
            )

            frames.append(
                image_to_bytes(
                    image
                )
            )

        capture.release()

        return frames, None

    except Exception as e:

        return [], str(e)

    finally:

        if temporary_file:

            try:
                os.remove(
                    temporary_file
                )
            except Exception:
                pass


# ============================================================
# AUDIO
# ============================================================

def transcribe_audio(
    audio_bytes,
):

    if not client:

        return (
            "[HF_TOKEN is missing.]"
        )

    try:

        result = (
            client.automatic_speech_recognition(
                audio_bytes,
                model=ASR_MODEL,
            )
        )

        if hasattr(
            result,
            "text",
        ):

            return result.text

        return str(result)

    except Exception as e:

        return (
            "[Audio transcription error] "
            + str(e)
        )


# ============================================================
# URL SECURITY
# ============================================================

def is_public_host(
    hostname,
):

    if not hostname:
        return False

    hostname = hostname.lower()

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

        for address in addresses:

            ip = address[4][0]

            parsed = (
                ipaddress.ip_address(
                    ip
                )
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

        parsed = urlparse(
            url
        )

        if parsed.scheme not in {
            "http",
            "https",
        }:
            return False

        if not parsed.hostname:
            return False

        return is_public_host(
            parsed.hostname
        )

    except Exception:

        return False


def find_urls(text):

    if not text:
        return []

    urls = re.findall(
        r"https?://[^\s<>\"]+",
        text,
        flags=re.IGNORECASE,
    )

    cleaned = []

    for url in urls:

        url = url.rstrip(
            ".,!?؛،)"
        )

        if url not in cleaned:
            cleaned.append(url)

    return cleaned


# ============================================================
# URL FETCHER
# ============================================================

def fetch_url(url):

    if not safe_url(url):

        return {
            "kind": "error",
            "text":
            "الرابط غير صالح أو محظور لأسباب أمنية.",
        }

    try:

        response = requests.get(
            url,
            timeout=20,
            stream=True,
            allow_redirects=True,
            headers={
                "User-Agent":
                "Mozilla/5.0 Mo-Dark-AI",
            },
        )

        response.raise_for_status()

        content_type = (
            response.headers.get(
                "content-type",
                "",
            ).lower()
        )

        content_length = (
            response.headers.get(
                "content-length"
            )
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
                        "الملف الموجود بالرابط كبير جداً.",
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
                    "تم تجاوز الحد الآمن لحجم الرابط.",
                }

            chunks.append(
                chunk
            )

        data = b"".join(
            chunks
        )

        # IMAGE
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

        # VIDEO
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

        # HTML
        if (
            "text/html"
            in content_type
        ):

            encoding = (
                response.encoding
                or "utf-8"
            )

            html = data.decode(
                encoding,
                errors="replace",
            )

            soup = BeautifulSoup(
                html,
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

            title = ""

            if soup.title:

                title = (
                    soup.title.get_text(
                        " ",
                        strip=True,
                    )
                )

            body = soup.get_text(
                "\n",
                strip=True,
            )

            body = body[:60000]

            return {
                "kind": "html",
                "text":
                f"""
URL:
{url}

PAGE TITLE:
{title}

PAGE CONTENT:
{body}
""",
            }

        # TEXT
        if content_type.startswith(
            "text/"
        ):

            text = data.decode(
                "utf-8",
                errors="replace",
            )

            return {
                "kind": "text",
                "text":
                text[:60000],
            }

        return {
            "kind": "binary",
            "text":
            f"""
URL:
{url}

CONTENT TYPE:
{content_type}

SIZE:
{len(data)} bytes
""",
        }

    except Exception as e:

        return {
            "kind": "error",
            "text":
            f"تعذر قراءة الرابط:\n{e}",
        }


# ============================================================
# PROCESS FILES
# ============================================================

def process_files(
    uploaded_files,
):

    text_context = []
    image_parts = []

    for file in uploaded_files:

        try:

            data = file.getvalue()

        except Exception:

            continue

        filename = file.name

        mime = (
            file.type
            or mimetypes.guess_type(
                filename
            )[0]
            or "application/octet-stream"
        )

        kind = detect_file_type(
            file
        )

        # IMAGE
        if kind == "image":

            image_parts.append(
                {
                    "type":
                    "image_url",
                    "image_url": {
                        "url":
                        data_url(
                            data,
                            mime,
                        )
                    },
                }
            )

            continue

        # VIDEO
        if kind == "video":

            frames, error = (
                extract_video_frames(
                    data
                )
            )

            text_context.append(
                f"""
VIDEO FILE:
{filename}

SIZE:
{len(data)} bytes

Sampled visual frames:
{len(frames)}
"""
            )

            for frame in frames:

                image_parts.append(
                    {
                        "type":
                        "image_url",
                        "image_url": {
                            "url":
                            data_url(
                                frame,
                                "image/png",
                            )
                        },
                    }
                )

            if error:

                text_context.append(
                    "Video processing note: "
                    + error
                )

            continue

        # AUDIO
        if kind == "audio":

            transcript = (
                transcribe_audio(
                    data
                )
            )

            text_context.append(
                f"""
AUDIO FILE:
{filename}

TRANSCRIPT:
{transcript}
"""
            )

            continue

        # PDF
        if kind == "pdf":

            text = extract_pdf(
                data
            )

            text_context.append(
                f"""
PDF FILE:
{filename}

CONTENT:
{text[:70000]}
"""
            )

            continue

        # DOCX
        if kind == "docx":

            text = extract_docx(
                data
            )

            text_context.append(
                f"""
DOCX FILE:
{filename}

CONTENT:
{text[:70000]}
"""
            )

            continue

        # EXCEL
        if kind == "excel":

            text = extract_excel(
                data
            )

            text_context.append(
                f"""
EXCEL FILE:
{filename}

DATA:
{text[:70000]}
"""
            )

            continue

        # CSV
        if kind == "csv":

            text = extract_csv(
                data
            )

            text_context.append(
                f"""
CSV FILE:
{filename}

DATA:
{text[:70000]}
"""
            )

            continue

        # TEXT / CODE
        if kind == "text":

            text = decode_bytes(
                data
            )

            text_context.append(
                f"""
TEXT / CODE FILE:
{filename}

CONTENT:
{text[:100000]}
"""
            )

            continue

        # ZIP
        if kind == "zip":

            text = inspect_zip(
                data
            )

            text_context.append(
                f"""
ZIP FILE:
{filename}

{text}
"""
            )

            continue

        # UNKNOWN
        text_context.append(
            f"""
BINARY FILE:
{filename}

MIME:
{mime}

SIZE:
{len(data)} bytes

This format was accepted but is not
directly parsed by the current application.
"""
        )

    return (
        "\n\n".join(
            text_context
        ),
        image_parts,
    )


# ============================================================
# NORMALIZE ARABIC / INTENT
# ============================================================

def normalize_text(text):

    if not text:
        return ""

    text = str(
        text
    ).strip().lower()

    replacements = {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ة": "ه",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
    }

    for old, new in replacements.items():

        text = text.replace(
            old,
            new,
        )

    # Remove Arabic diacritics
    text = re.sub(
        r"[\u064B-\u065F\u0670]",
        "",
        text,
    )

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ============================================================
# IMAGE INTENT
# ============================================================

def wants_image_generation(
    text,
):

    t = normalize_text(
        text
    )

    patterns = [
        r"انشئ.*صوره",
        r"انشاء.*صوره",
        r"انشئلي.*صوره",
        r"انشاءلي.*صوره",
        r"سوي.*صوره",
        r"سويلي.*صوره",
        r"صمم.*صوره",
        r"صمملي.*صوره",
        r"ارسم.*صوره",
        r"ارسملي.*صوره",
        r"ولد.*صوره",
        r"ولّد.*صوره",
        r"توليد.*صوره",
        r"اعمل.*صوره",
        r"اعمللي.*صوره",
        r"اصنع.*صوره",
        r"اصنعلي.*صوره",
    ]

    for pattern in patterns:
        if re.search(pattern, t):
            return True

    return False


# ============================================================
# STREAMLIT UI RUNNER & MAIN CHAT INPUT
# ============================================================

uploaded_files = st.file_uploader(
    "ارفع ملفات (صور، مستندات، أكواد، صوت، فيديو)",
    accept_multiple_files=True,
    key="file_uploader_box"
)

user_query = st.chat_input("اطرح سؤالك أو اطلب إنشاء صورة / تحليل ملف...")

messages = load_messages(st.session_state.session_id)

for msg in messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_query or uploaded_files:

    files_text, files_images = process_files(uploaded_files or [])
    
    full_prompt = user_query or ""
    if files_text:
        full_prompt = f"{full_prompt}\n\n[Uploaded Files Context]:\n{files_text}"

    url_list = find_urls(user_query or "")
    for u in url_list:
        fetched = fetch_url(u)
        if fetched["kind"] != "error":
            full_prompt += f"\n\nhttps://u.co.uk/:\n{fetched['text']}"

    if user_query:
        save_message(st.session_state.session_id, "user", user_query)
        with st.chat_message("user"):
            st.markdown(user_query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # Image Generation Logic
        if wants_image_generation(user_query or ""):
            response_placeholder.text("جاري توليد الصورة عبر FLUX...")
            try:
                image_result = client.text_to_image(
                    prompt=user_query,
                    model=IMAGE_MODEL,
                )
                st.image(image_result, caption="النتيجة المולدة", use_container_width=True)
                
                # Save as image bytes to message
                buffered = io.BytesIO()
                image_result.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                img_md = f"data:image/png;base64,{img_str}"
                
                save_message(st.session_state.session_id, "assistant", f"![Generated Image]({img_md})")
            except Exception as e:
                response_placeholder.error(f"حدث خطأ أثناء توليد الصورة: {e}")
        else:
            # Standard Text / Vision Generation Logic
            try:
                messages_payload = [{"role": "user", "content": full_prompt}]
                
                if files_images and client:
                    # Vision request
                    content_list = [{"type": "text", "text": full_prompt}] + files_images
                    messages_payload[0]["content"] = content_list
                    chat_completion = client.chat.completions.create(
                        model=VISION_MODEL,
                        messages=messages_payload,
                        max_tokens=2048,
                    )
                else:
                    chat_completion = client.chat.completions.create(
                        model=TEXT_MODEL,
                        messages=messages_payload,
                        max_tokens=2048,
                    )
                
                output_text = chat_completion.choices[0].message.content
                response_placeholder.markdown(output_text)
                save_message(st.session_state.session_id, "assistant", output_text)
                
            except Exception as e:
                response_placeholder.error(f"خطأ في الاتصال بالنموذج: {e}")
