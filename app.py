import streamlit as st
from huggingface_hub import InferenceClient

# ==============================
# MO DARK AI
# ==============================

st.set_page_config(
    page_title="Mo Dark AI",
    page_icon="🖤",
    layout="wide"
)

# ==============================
# STYLE
# ==============================

st.markdown("""
<style>
.stApp {
    background: #080808;
    color: #eeeeee;
}

.block-container {
    max-width: 1200px;
    padding-top: 3rem;
}

h1 {
    font-size: 52px !important;
    font-weight: 800 !important;
}

.mo-subtitle {
    color: #888;
    font-size: 18px;
    margin-bottom: 30px;
}

[data-testid="stChatMessage"] {
    border-radius: 18px;
    padding: 12px;
}

.stChatInput textarea {
    background: #111 !important;
    color: white !important;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# HEADER
# ==============================

st.title("🖤 Mo Dark AI")

st.markdown(
    '<div class="mo-subtitle">'
    'AI Developer • Coding Assistant • Project Builder'
    '</div>',
    unsafe_allow_html=True
)

# ==============================
# HF CLIENT
# ==============================

@st.cache_resource
def get_client():
    return InferenceClient(
        api_key=st.secrets["HF_TOKEN"]
    )

client = get_client()

# ==============================
# MODEL
# ==============================

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# ==============================
# SYSTEM PROMPT
# ==============================

SYSTEM_PROMPT = """
You are Mo Dark AI, an advanced AI software engineer.

Your main job is programming, software engineering, debugging,
architecture, automation, and building complete projects.

IMPORTANT RULES:

1. Understand the user's request before answering.
2. Give complete solutions, not tiny fragments.
3. When the user asks for code, provide complete working code.
4. Never intentionally shorten code just to make the answer shorter.
5. If a project contains multiple files, clearly separate every file.
6. Explain where each file belongs.
7. Preserve existing functionality when modifying code.
8. Carefully inspect code for syntax errors.
9. Think about dependencies and imports.
10. Prefer robust, maintainable solutions.
11. When debugging, identify the actual cause of the error.
12. Give the corrected code, not only an explanation.
13. Support Python, JavaScript, HTML, CSS, C, C++, Java,
    SQL, Bash, PowerShell and other common programming languages.
14. If the user asks to create a tool, build the actual tool.
15. If the request is ambiguous, make a reasonable engineering
    assumption and state it briefly.
16. Do not replace a complete solution with pseudocode unless
    the user explicitly asks for pseudocode.
17. For large projects, organize the response into files and modules.
18. Be especially careful with Streamlit applications.
19. Respond in the user's language when possible.
20. The user may speak Iraqi Arabic. Understand Iraqi Arabic naturally.

You are not merely a chatbot.
Act like a senior software engineer helping the user build real projects.
"""

# ==============================
# CHAT MEMORY
# ==============================

if "messages" not in st.session_state:
    st.session_state.messages = []

# ==============================
# DISPLAY HISTORY
# ==============================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ==============================
# USER INPUT
# ==============================

prompt = st.chat_input(
    "اكتب لـ Mo Dark شتريد يسويلك..."
)

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(st.session_state.messages)

    with st.chat_message("assistant"):

        response_box = st.empty()

        try:

            response = client.chat_completion(
                model=MODEL,
                messages=messages,
                max_tokens=8192,
                temperature=0.15,
            )

            answer = response.choices[0].message.content

            response_box.markdown(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            error_message = (
                "⚠️ صار خطأ أثناء الاتصال بالذكاء الاصطناعي:\n\n"
                f"`{str(e)}`"
            )

            response_box.error(error_message)
