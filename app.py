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

    # IMPORTANT:
    # These patterns are intentionally broad
    # so "انشاء صورة بنت جميلة" works.

    patterns = [

        # Arabic
        r"انشئ.*صوره",
        r"انشاء.*صوره",
        r"انشئلي.*صوره",
        r"انشاءلي.*صوره",
        r"سوي.*صوره",
        r"سويلي.*صوره",
        r"سوي.*صوره",
        r"صمم.*صوره",
        r"صمملي.*صوره",
        r"صمم لي.*صوره",
        r"ارسم.*صوره",
        r"ارسملي.*صوره",
        r"ولد.*صوره",
        r"ولّد.*صوره",
        r"توليد.*صوره",
        r"اعمل.*صوره",
        r"اعمللي.*صوره",
        r"اصنع.*صوره",
        r"اصنعلي.*صوره",

        # English
        r"generate.*image",
        r"create.*image",
        r"make.*image",
        r"draw.*image",
        r"generate.*picture",
        r"create.*picture",
        r"make.*picture",
        r"draw.*picture",
        r"text to image",
        r"text-to-image",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            t,
        ):
            return True

    # Direct phrases
    direct = [

        "انشئ صوره",
        "انشاء صوره",
        "انشئلي صوره",
        "سوي صوره",
        "سويلي صوره",
        "صمم صوره",
        "صمملي صوره",
        "ارسم صوره",
        "توليد صوره",
        "اصنع صوره",
        "اعمل صوره",

        "generate image",
        "create image",
        "make image",
        "draw image",
        "generate picture",
        "create picture",
        "make a picture",
    ]

    return any(
        phrase in t
        for phrase in direct
    )


# ============================================================
# VIDEO INTENT
# ============================================================

def wants_video_generation(
    text,
):

    t = normalize_text(
        text
    )

    patterns = [

        # Arabic
        r"انشئ.*فيديو",
        r"انشاء.*فيديو",
        r"انشئلي.*فيديو",
        r"سوي.*فيديو",
        r"سويلي.*فيديو",
        r"صمم.*فيديو",
        r"صمملي.*فيديو",
        r"ولد.*فيديو",
        r"توليد.*فيديو",
        r"اعمل.*فيديو",
        r"اعمللي.*فيديو",
        r"اصنع.*فيديو",
        r"اصنعلي.*فيديو",

        # Image -> Video
        r"حول.*صوره.*فيديو",
        r"حول.*الصوره.*فيديو",
        r"حولها.*فيديو",

        # English
        r"generate.*video",
        r"create.*video",
        r"make.*video",
        r"text to video",
        r"text-to-video",
        r"image to video",
        r"image-to-video",
    ]

    for pattern in patterns:

        if re.search(
            pattern,
            t,
        ):
            return True

    return any(
        phrase in t
        for phrase in [
            "انشئ فيديو",
            "انشاء فيديو",
            "سوي فيديو",
            "سويلي فيديو",
            "صمم فيديو",
            "توليد فيديو",
            "اصنع فيديو",
            "حولها لفيديو",
            "حول الصورة الى فيديو",
            "generate video",
            "create video",
            "make video",
            "image to video",
        ]
    )


# ============================================================
# IMAGE EDIT
# ============================================================

def wants_image_edit(
    text,
):

    t = normalize_text(
        text
    )

    phrases = [

        "عدل الصورة",
        "عدل الصوره",
        "تعديل الصورة",
        "تعديل الصوره",
        "غير الصورة",
        "غير الصوره",
        "غيّر الصورة",
        "حسن الصورة",
        "حسن الصوره",
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


# ============================================================
# IMAGE FILE
# ============================================================

def has_image_file(
    files,
):

    for file in files:

        if (
            detect_file_type(file)
            == "image"
        ):
            return True

    return False


def first_image_file(
    files,
):

    for file in files:

        if (
            detect_file_type(file)
            == "image"
        ):
            return file

    return None


# ============================================================
# GENERATE IMAGE
# ============================================================

def generate_image(
    prompt,
):

    if not client:

        raise RuntimeError(
            "HF_TOKEN غير موجود في Streamlit Secrets."
        )

    image = client.text_to_image(
        prompt=prompt,
        model=IMAGE_MODEL,
        width=IMAGE_WIDTH,
        height=IMAGE_HEIGHT,
    )

    return image_to_bytes(
        image
    )


# ============================================================
# GENERATE VIDEO
# ============================================================

def generate_video(
    prompt,
):

    if not client:

        raise RuntimeError(
            "HF_TOKEN غير موجود في Streamlit Secrets."
        )

    result = client.text_to_video(
        prompt=prompt,
        model=VIDEO_MODEL,
    )

    return bytes(
        result
    )


# ============================================================
# IMAGE -> VIDEO
# ============================================================

def convert_image_to_video(
    image_bytes,
    prompt,
):

    if not client:

        raise RuntimeError(
            "HF_TOKEN غير موجود في Streamlit Secrets."
        )

    result = client.image_to_video(
        image=image_bytes,
        prompt=prompt,
        model=IMAGE_VIDEO_MODEL,
    )

    return bytes(
        result
    )


# ============================================================
# EDIT IMAGE
# ============================================================

def edit_image(
    image_bytes,
    prompt,
):

    if not client:

        raise RuntimeError(
            "HF_TOKEN غير موجود في Streamlit Secrets."
        )

    result = client.image_to_image(
        image=image_bytes,
        prompt=prompt,
        model=IMAGE_MODEL,
    )

    return image_to_bytes(
        result
    )


# ============================================================
# AI CHAT
# ============================================================

SYSTEM_PROMPT = """
You are Mo Dark AI.

You are a multimodal AI assistant.

Your capabilities include:

Programming
Software engineering
Python
Streamlit
JavaScript
HTML
CSS
APIs
Databases
Debugging
Project architecture
Data analysis
Documents
Images
Videos
URLs
Audio transcripts
General questions
Writing
Advice
Research-style reasoning

IMPORTANT:

- Follow the user's exact request.
- Do not change the requested framework.
- If the user asks for Streamlit, use Streamlit.
- If the user asks for complete code, provide complete code.
- Do not intentionally omit important files.
- Keep multi-file projects consistent.
- Check imports and dependencies logically.
- Never claim you executed code when you did not.
- If visual content is provided, analyze it.
- If video frames are provided, analyze the sequence.
- If documents are provided, use their extracted contents.
- If a URL is provided, use the retrieved URL content.
- Do not invent information.
- Answer in the user's language when appropriate.
- Be practical and direct.

The application itself handles image generation,
video generation and image-to-video before reaching you.

Therefore, when the user is asking for generation,
do not tell them that you cannot generate media.
"""


def chat_with_ai(
    prompt,
    history,
    text_context="",
    image_parts=None,
):

    if not client:

        raise RuntimeError(
            "HF_TOKEN غير موجود. "
            "أضفه في Streamlit Secrets."
        )

    image_parts = (
        image_parts
        or []
    )

    is_vision = bool(
        image_parts
    )

    model = (
        VISION_MODEL
        if is_vision
        else TEXT_MODEL
    )

    messages = [
        {
            "role": "system",
            "content":
            SYSTEM_PROMPT,
        }
    ]

    for item in history[-12:]:

        messages.append(
            {
                "role":
                item["role"],
                "content":
                item["content"],
            }
        )

    context = ""

    if text_context:

        context = (
            "\n\n"
            "ATTACHED / RETRIEVED CONTEXT:\n"
            + text_context
        )

    if is_vision:

        content = [
            {
                "type": "text",
                "text":
                prompt
                + context
                + "\n\n"
                "Analyze the supplied visual content "
                "carefully.",
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

        messages.append(
            {
                "role": "user",
                "content":
                prompt + context,
            }
        )

    try:

        response = (
            client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=8192,
                temperature=0.12,
            )
        )

    except Exception as first_error:

        if is_vision:

            response = (
                client.chat_completion(
                    model=VISION_FALLBACK,
                    messages=messages,
                    max_tokens=8192,
                    temperature=0.12,
                )
            )

        else:

            raise first_error

    try:

        return (
            response
            .choices[0]
            .message
            .content
        )

    except Exception:

        return str(
            response
        )


# ============================================================
# FILE PREVIEW
# ============================================================

def show_file_preview(
    uploaded_file,
):

    kind = detect_file_type(
        uploaded_file
    )

    if kind == "image":

        try:

            uploaded_file.seek(0)

            st.image(
                uploaded_file,
                caption=uploaded_file.name,
                width="stretch",
            )

        except Exception:

            st.caption(
                "🖼️ "
                + uploaded_file.name
            )

    elif kind == "video":

        try:

            uploaded_file.seek(0)

            st.video(
                uploaded_file
            )

        except Exception:

            st.caption(
                "🎬 "
                + uploaded_file.name
            )

    elif kind == "audio":

        try:

            uploaded_file.seek(0)

            st.audio(
                uploaded_file
            )

        except Exception:

            st.caption(
                "🎵 "
                + uploaded_file.name
            )

    else:

        st.caption(
            "📎 "
            + uploaded_file.name
        )


# ============================================================
# CURRENT CHAT
# ============================================================

current_messages = load_messages(
    st.session_state.session_id
)


# ============================================================
# WELCOME
# ============================================================

if not current_messages:

    render_html(
        """
<div style="
    max-width:850px;
    margin:55px auto 40px auto;
    text-align:center;
">

    <div style="
        width:78px;
        height:78px;
        margin:auto;
        border-radius:24px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-size:36px;

        background:
            linear-gradient(
                135deg,
                #00eaff,
                #2878ff,
                #7c3cff
            );

        box-shadow:
            0 0 55px
            rgba(0,234,255,.20);
    ">
        ⚡
    </div>

    <div style="
        font-size:43px;
        line-height:1.1;
        font-weight:800;
        letter-spacing:-2px;
        margin-top:25px;
    ">
        What can I build
        <span style="
            color:#00eaff;
        ">
            for you?
        </span>
    </div>

    <div style="
        max-width:680px;
        margin:18px auto 0 auto;
        color:#7d8798;
        font-size:14px;
        line-height:1.9;
    ">
        اكتب طلبك فقط.
        لا تحتاج تختار وضع أو أداة.
        Mo Dark AI يفهم تلقائياً إذا تريد
        برمجة، صورة، فيديو، تحليل ملف،
        تحليل رابط أو محادثة عادية.
    </div>

    <div style="
        display:flex;
        justify-content:center;
        flex-wrap:wrap;
        gap:8px;
        margin-top:25px;
    ">

        <span style="
            padding:8px 13px;
            border-radius:999px;
            border:1px solid rgba(0,234,255,.13);
            color:#7d8798;
            font-size:11px;
        ">
            💻 Programming
        </span>

        <span style="
            padding:8px 13px;
            border-radius:999px;
            border:1px solid rgba(0,234,255,.13);
            color:#7d8798;
            font-size:11px;
        ">
            🖼️ Images
        </span>

        <span style="
            padding:8px 13px;
            border-radius:999px;
            border:1px solid rgba(0,234,255,.13);
            color:#7d8798;
            font-size:11px;
        ">
            🎬 Videos
        </span>

        <span style="
            padding:8px 13px;
            border-radius:999px;
            border:1px solid rgba(0,234,255,.13);
            color:#7d8798;
            font-size:11px;
        ">
            📁 Files
        </span>

        <span style="
            padding:8px 13px;
            border-radius:999px;
            border:1px solid rgba(0,234,255,.13);
            color:#7d8798;
            font-size:11px;
        ">
            🌐 URLs
        </span>

    </div>

</div>
"""
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in current_messages:

    role = message["role"]

    content = message["content"]

    if role not in {
        "user",
        "assistant",
    }:
        continue

    with st.chat_message(
        role
    ):

        st.markdown(
            content
        )


# ============================================================
# CHAT INPUT
# ============================================================

try:

    chat_value = st.chat_input(
        "اكتب أي شيء... صورة، فيديو، كود، ملف، رابط، سؤال...",
        accept_file="multiple",
        file_type=None,
        max_upload_size=2048,
        key="main_chat_input",
    )

except TypeError:

    chat_value = st.chat_input(
        "اكتب أي شيء...",
        accept_file="multiple",
        file_type=None,
        key="main_chat_input",
    )


# ============================================================
# HANDLE USER REQUEST
# ============================================================

if chat_value:

    # New Streamlit ChatInputValue
    if isinstance(
        chat_value,
        str,
    ):

        user_prompt = (
            chat_value.strip()
        )

        uploaded_files = []

    else:

        user_prompt = (
            getattr(
                chat_value,
                "text",
                "",
            )
            or ""
        ).strip()

        uploaded_files = list(
            getattr(
                chat_value,
                "files",
                [],
            )
            or []
        )

    # ------------------------------------------
    # If files only
    # ------------------------------------------

    if (
        not user_prompt
        and uploaded_files
    ):

        user_prompt = (
            "حلل الملفات المرفقة بالتفصيل "
            "واشرح لي محتواها وأهم المعلومات "
            "الموجودة فيها."
        )

    if not user_prompt:

        st.stop()

    # ------------------------------------------
    # Title
    # ------------------------------------------

    old_messages = load_messages(
        st.session_state.session_id
    )

    if not old_messages:

        title = (
            user_prompt
            .replace("\n", " ")
            .strip()
        )

        if len(title) > 55:

            title = (
                title[:55]
                + "..."
            )

        update_session_title(
            st.session_state.session_id,
            title,
        )

    # ------------------------------------------
    # Save user
    # ------------------------------------------

    filenames = [
        file.name
        for file in uploaded_files
    ]

    saved_user_text = user_prompt

    if filenames:

        saved_user_text += (
            "\n\n📎 الملفات: "
            + ", ".join(
                filenames
            )
        )

    save_message(
        st.session_state.session_id,
        "user",
        saved_user_text,
    )

    # ------------------------------------------
    # Display user
    # ------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            user_prompt
        )

        if uploaded_files:

            for file in uploaded_files:

                with st.expander(
                    "📎 "
                    + file.name,
                    expanded=False,
                ):

                    show_file_preview(
                        file
                    )

    # ------------------------------------------
    # AI
    # ------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        try:

            with st.spinner(
                "Mo Dark AI يعمل..."
            ):

                # ==================================
                # PROCESS FILES
                # ==================================

                text_context = ""
                image_parts = []

                if uploaded_files:

                    (
                        text_context,
                        image_parts,
                    ) = process_files(
                        uploaded_files
                    )

                # ==================================
                # PROCESS URLS
                # ==================================

                urls = find_urls(
                    user_prompt
                )

                url_context = []
                url_image_parts = []

                for url in urls[:3]:

                    result = fetch_url(
                        url
                    )

                    kind = result.get(
                        "kind"
                    )

                    if kind == "image":

                        url_image_parts.append(
                            {
                                "type":
                                "image_url",
                                "image_url": {
                                    "url":
                                    data_url(
                                        result[
                                            "data"
                                        ],
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
                                result[
                                    "data"
                                ]
                            )
                        )

                        url_context.append(
                            f"""
VIDEO URL:
{url}

Sampled frames:
{len(frames)}
"""
                        )

                        for frame in frames:

                            url_image_parts.append(
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

                            url_context.append(
                                "Video note: "
                                + error
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
                    url_image_parts
                )

                # ==================================
                # AUTO ROUTER
                # ==================================

                image_attached = (
                    has_image_file(
                        uploaded_files
                    )
                )

                image_file = (
                    first_image_file(
                        uploaded_files
                    )
                )

                # ==================================
                # 1. VIDEO
                # ==================================

                if wants_video_generation(
                    user_prompt
                ):

                    # Image -> Video
                    if (
                        image_attached
                        and image_file
                    ):

                        image_bytes = (
                            image_file.getvalue()
                        )

                        status = st.empty()

                        status.info(
                            "🎬 جاري تحويل الصورة إلى فيديو..."
                        )

                        video_bytes = (
                            convert_image_to_video(
                                image_bytes,
                                user_prompt,
                            )
                        )

                        status.empty()

                        st.success(
                            "🎬 تم إنشاء الفيديو"
                        )

                        st.video(
                            video_bytes
                        )

                        st.download_button(
                            "⬇️ تنزيل الفيديو",
                            data=video_bytes,
                            file_name=(
                                "mo_dark_video.mp4"
                            ),
                            mime="video/mp4",
                            use_container_width=True,
                        )

                        answer = (
                            "🎬 تم إنشاء الفيديو "
                            "من الصورة حسب طلبك."
                        )

                    # Text -> Video
                    else:

                        status = st.empty()

                        status.info(
                            "🎬 جاري إنشاء الفيديو..."
                        )

                        video_bytes = (
                            generate_video(
                                user_prompt
                            )
                        )

                        status.empty()

                        st.success(
                            "🎬 تم إنشاء الفيديو"
                        )

                        st.video(
                            video_bytes
                        )

                        st.download_button(
                            "⬇️ تنزيل الفيديو",
                            data=video_bytes,
                            file_name=(
                                "mo_dark_video.mp4"
                            ),
                            mime="video/mp4",
                            use_container_width=True,
                        )

                        answer = (
                            "🎬 تم إنشاء الفيديو "
                            "حسب طلبك."
                        )

                # ==================================
                # 2. IMAGE EDIT
                # ==================================

                elif (
                    image_attached
                    and image_file
                    and wants_image_edit(
                        user_prompt
                    )
                ):

                    status = st.empty()

                    status.info(
                        "🖼️ جاري تعديل الصورة..."
                    )

                    output = edit_image(
                        image_file.getvalue(),
                        user_prompt,
                    )

                    status.empty()

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
                        file_name=(
                            "mo_dark_edited.png"
                        ),
                        mime="image/png",
                        use_container_width=True,
                    )

                    answer = (
                        "🖼️ تم تعديل الصورة "
                        "حسب طلبك."
                    )

                # ==================================
                # 3. IMAGE GENERATION
                # ==================================

                elif wants_image_generation(
                    user_prompt
                ):

                    status = st.empty()

                    status.info(
                        "🖼️ جاري إنشاء الصورة..."
                    )

                    output = generate_image(
                        user_prompt
                    )

                    status.empty()

                    st.success(
                        "🖼️ تم إنشاء الصورة"
                    )

                    st.image(
                        output,
                        width="stretch",
                    )

                    st.download_button(
                        "⬇️ تنزيل الصورة",
                        data=output,
                        file_name=(
                            "mo_dark_image.png"
                        ),
                        mime="image/png",
                        use_container_width=True,
                    )

                    answer = (
                        "🖼️ تم إنشاء الصورة "
                        "بالمقاس القياسي "
                        "1024×1024."
                    )

                # ==================================
                # 4. NORMAL CHAT / VISION
                # ==================================

                else:

                    rows = load_messages(
                        st.session_state.session_id
                    )

                    history = []

                    for row in rows:

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

                    # Current user is already stored,
                    # but we pass it separately.
                    if (
                        history
                        and history[-1][
                            "role"
                        ] == "user"
                    ):

                        history = history[:-1]

                    answer = chat_with_ai(
                        prompt=user_prompt,
                        history=history,
                        text_context=text_context,
                        image_parts=image_parts,
                    )

                    st.markdown(
                        answer
                    )

                # ==================================
                # SAVE ANSWER
                # ==================================

                save_message(
                    st.session_state.session_id,
                    "assistant",
                    answer,
                )

        except Exception as error:

            error_text = str(
                error
            )

            error_message = (
                "❌ صار خطأ أثناء تنفيذ الطلب.\n\n"
                "```text\n"
                + error_text
                + "\n```\n\n"
                "إذا كان الطلب إنشاء صورة أو فيديو، "
                "فقد يكون نموذج التوليد غير متاح "
                "حالياً عبر Hugging Face أو انتهت "
                "الحصة المجانية المتاحة."
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

render_html(
    """
<div style="
    position:fixed;
    left:0;
    right:0;
    bottom:8px;
    text-align:center;
    pointer-events:none;
    z-index:1;
">

    <span style="
        color:#343d4d;
        font-size:9px;
        letter-spacing:1px;
    ">
        MO DARK AI • MULTIMODAL INTELLIGENCE
    </span>

</div>
"""
)
