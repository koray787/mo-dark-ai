import streamlit as st
from huggingface_hub import InferenceClient

# =========================================================
# MO DARK AI — ADVANCED CODING ASSISTANT
# =========================================================

st.set_page_config(
    page_title="Mo Dark AI",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# STYLE
# =========================================================

st.markdown("""
<style>
.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(80,80,80,.10), transparent 30%),
        radial-gradient(circle at 90% 90%, rgba(80,80,80,.08), transparent 30%),
        #070707;
    color: #f5f5f5;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 5rem;
}

h1 {
    font-size: 48px !important;
    font-weight: 900 !important;
    letter-spacing: -1px;
}

.mo-subtitle {
    color: #999;
    font-size: 17px;
    margin-top: -18px;
    margin-bottom: 30px;
}

[data-testid="stChatMessage"] {
    border: 1px solid #202020;
    border-radius: 18px;
    padding: 14px;
    margin-bottom: 10px;
    background: rgba(15,15,15,.75);
}

[data-testid="stSidebar"] {
    background: #0b0b0b;
    border-right: 1px solid #202020;
}

.stChatInput textarea {
    background: #111 !important;
    color: white !important;
}

.stButton button {
    border-radius: 10px;
}

.mo-status {
    padding: 12px 15px;
    border: 1px solid #252525;
    border-radius: 12px;
    background: #101010;
    color: #aaa;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are Mo Dark AI, a senior software engineer and expert coding assistant.

Your job is to help the user build REAL software and solve programming
problems professionally.

CORE RULES:

1. Understand the request before answering.
2. When the user asks for code, provide complete usable code.
3. Never intentionally replace a requested complete program with pseudocode.
4. Never omit important sections of code just to make the response shorter.
5. If the project requires multiple files, provide every required file.
6. Clearly label each file with its filename.
7. Explain where each file belongs.
8. Preserve existing functionality when modifying code.
9. Do not randomly rewrite working parts of a project.
10. Carefully check imports, syntax, variables, indentation and dependencies.
11. When debugging an error, identify the likely cause and provide the corrected code.
12. If information is missing, make a reasonable engineering assumption.
13. Prefer robust and maintainable implementations.
14. Use modern library APIs when possible.
15. Avoid outdated package versions unless specifically requested.
16. For Streamlit projects, make sure Streamlit-specific code is valid.
17. If a function uses a library, make sure that library is imported.
18. If a project needs requirements.txt, include it.
19. If a project needs configuration files, include them.
20. If the user asks for a website, provide the actual HTML/CSS/JS needed.
21. If the user asks for an application, provide the actual implementation.
22. If the user gives an error message, analyze the error directly.
23. Do not claim that code was executed or tested when it was not.
24. Tell the user what they need to install or configure.
25. Support Python, JavaScript, HTML, CSS, C, C++, Java, SQL,
    Bash, PowerShell and other common programming languages.
26. Understand Iraqi Arabic naturally.
27. Reply in the user's language when possible.
28. Keep explanations clear, but prioritize the actual solution.
29. For large projects, organize the solution into logical files/modules.
30. Think like a senior developer reviewing production code.

IMPORTANT FOR CODE:

Before giving code, mentally check:
- imports
- syntax
- indentation
- undefined variables
- missing functions
- missing dependencies
- incorrect library usage
- obvious runtime errors
- file relationships

Do not intentionally give incomplete code when the user requested a complete program.

You are Mo Dark AI.
You are a coding-focused AI assistant designed to help build real projects.
"""

# =========================================================
# HUGGING FACE
# =========================================================

@st.cache_resource
def get_client():
    return InferenceClient(
        api_key=st.secrets["HF_TOKEN"]
    )


try:
    client = get_client()
    connection_ok = True
except Exception:
    client = None
    connection_ok = False


MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"

# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("## 🖤 Mo Dark AI")

    st.caption("Advanced AI Coding Assistant")

    st.divider()

    if connection_ok:
        st.success("AI متصل")
    else:
        st.error("AI غير متصل")

    st.markdown("### قدرات النظام")

    st.markdown("""
    🧠 فهم البرمجة  
    💻 كتابة الأكواد  
    🐛 تصحيح الأخطاء  
    🏗️ بناء المشاريع  
    📁 مشاريع متعددة الملفات  
    🌐 Web Development  
    🐍 Python  
    ⚡ JavaScript  
    🗄️ SQL  
    🎨 HTML / CSS
    """)

    st.divider()

    if st.button("🗑️ مسح المحادثة", use_container_width=True):

        st.session_state.messages = []

        st.rerun()

# =========================================================
# HEADER
# =========================================================

st.title("🖤 Mo Dark AI")

st.markdown(
    '<div class="mo-subtitle">'
    'Senior AI Software Engineer • Coding • Debugging • Project Building'
    '</div>',
    unsafe_allow_html=True
)

if connection_ok:

    st.markdown(
        '<div class="mo-status">'
        '🟢 النظام جاهز — اكتب طلبك البرمجي بالأسفل'
        '</div>',
        unsafe_allow_html=True
    )

else:

    st.error(
        "تعذر الاتصال بالذكاء الاصطناعي. "
        "تأكد من وجود HF_TOKEN داخل Secrets."
    )

# =========================================================
# EXAMPLE PROMPTS
# =========================================================

if not st.session_state.messages:

    st.markdown("### 🚀 جرّب مثلاً")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(
            "🐍 اكتبلي برنامج Python كامل "
            "لإدارة المخزون"
        )

    with col2:
        st.info(
            "🌐 سويلي موقع HTML CSS JavaScript "
            "احترافي"
        )

    with col3:
        st.info(
            "🐛 هذا الكود بيه خطأ، أصلحه "
            "وفسرلي السبب"
        )

# =========================================================
# DISPLAY CHAT HISTORY
# =========================================================

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    with st.chat_message(role):

        st.markdown(content)

# =========================================================
# USER INPUT
# =========================================================

prompt = st.chat_input(
    "اكتب لـ Mo Dark شتريد يسويلك..."
)

# =========================================================
# PROCESS REQUEST
# =========================================================

if prompt:

    # -----------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    # -----------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)

    # -----------------------------------------
    # PREPARE MESSAGES
    # -----------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        }
    ]

    messages.extend(st.session_state.messages)

    # -----------------------------------------
    # AI RESPONSE
    # -----------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        try:

            response = client.chat_completion(
                model=MODEL,
                messages=messages,
                max_tokens=8192,
                temperature=0.15,
            )

            answer = response.choices[0].message.content

            if not answer:
                answer = "ما حصلت على إجابة من النموذج."

            response_placeholder.markdown(answer)

            # -----------------------------------------
            # SAVE AI RESPONSE
            # -----------------------------------------

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

        except Exception as e:

            error_text = str(e)

            response_placeholder.error(
                "⚠️ صار خطأ أثناء تشغيل الذكاء الاصطناعي."
            )

            st.code(
                error_text,
                language="text"
            )
