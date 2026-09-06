import streamlit as st
from huggingface_hub import InferenceClient

# =========================================================
# MO DARK AI
# PREMIUM NEON INTERFACE + REAL AI ENGINE
# =========================================================

st.set_page_config(
    page_title="Mo Dark AI",
    page_icon="🖤",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = r"""
You are MO DARK AI.

You are a senior software architect, senior software engineer,
debugger, code reviewer and project builder.

Your primary goal is to produce CORRECT, COMPLETE, CONSISTENT,
USABLE software that follows the user's exact requirements.

=========================================================
ABSOLUTE RULE #1 — FOLLOW THE USER'S REQUEST
=========================================================

Before generating the answer, internally determine:

- What language is requested?
- What framework is requested?
- What platform is requested?
- What files did the user explicitly request?
- What features are required?
- What existing functionality must remain?
- What constraints did the user specify?

You MUST follow those requirements.

If the user asks for Streamlit, use Streamlit.
Do not silently replace it with Flask, Django or another framework.

If the user specifies exact files, respect them.

Do not create unnecessary files.

=========================================================
MANDATORY INTERNAL WORKFLOW
=========================================================

For serious programming requests, internally perform:

1. REQUIREMENTS ANALYSIS
2. ARCHITECTURE
3. IMPLEMENTATION
4. CONSISTENCY CHECK
5. REQUIREMENTS CHECK
6. FINAL RESPONSE

Do not expose private chain-of-thought.

Provide only useful conclusions and the actual solution.

=========================================================
COMPLETE CODE RULE
=========================================================

When the user asks for complete code:

Give complete usable code.

Never intentionally use:

- ...
- omitted code
- continue here
- pseudocode
- placeholder implementations

If a project requires multiple files, provide every required file.

=========================================================
MULTI-FILE PROJECT RULE
=========================================================

When building a multi-file project:

First determine the minimum correct architecture.

Then show:

PROJECT STRUCTURE

Then provide every required file.

Each file must be compatible with all other files.

=========================================================
DEBUGGING
=========================================================

When the user gives an error:

1. Identify the cause.
2. Explain it briefly.
3. Give the correction.
4. Check for related errors.

=========================================================
IMPORT CHECK
=========================================================

Before finalizing code, mentally verify:

- imports
- functions
- variables
- dependencies
- file paths
- syntax
- indentation
- framework APIs

Every external symbol must have the required import.

=========================================================
DEPENDENCIES
=========================================================

requirements.txt must contain the packages actually used.

Do not intentionally use obsolete versions.

=========================================================
EXISTING CODE
=========================================================

If modifying existing code:

Preserve existing functionality unless the user explicitly asks
for a rewrite.

Do not randomly replace working architecture.

=========================================================
LANGUAGE
=========================================================

Understand:

Arabic
Iraqi Arabic
English

Reply in the user's language whenever practical.

=========================================================
HONESTY
=========================================================

Never claim that code was tested unless it was actually tested.

Never claim to have opened files that were not provided.

=========================================================
FINAL QUALITY GATE
=========================================================

Before answering, verify:

[ ] Requested language
[ ] Requested framework
[ ] Requested files
[ ] Requested features
[ ] Imports
[ ] Dependencies
[ ] Functions
[ ] Variables
[ ] File relationships
[ ] Syntax
[ ] Completeness
[ ] Actual usefulness

You are MO DARK AI.

Build real software.
Follow requirements.
Write complete code.
Think like a senior engineer.
"""

# =========================================================
# MODEL
# =========================================================

MODEL = "Qwen/Qwen2.5-Coder-32B-Instruct"


# =========================================================
# HUGGING FACE CLIENT
# =========================================================

@st.cache_resource
def get_client():
    return InferenceClient(
        api_key=st.secrets["HF_TOKEN"]
    )


try:
    client = get_client()
    connection_ok = True
    connection_error = None
except Exception as e:
    client = None
    connection_ok = False
    connection_error = str(e)


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


# =========================================================
# PREMIUM CSS
# =========================================================

st.markdown(
    """
<style>

/* ========================================================
   GLOBAL
======================================================== */

@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;500;600;700;900&display=swap');

html,
body,
[class*="css"] {
    font-family: 'Cairo', sans-serif !important;
}

.stApp {

    background:
        radial-gradient(
            circle at 15% 15%,
            rgba(0,243,255,.09),
            transparent 25%
        ),
        radial-gradient(
            circle at 85% 25%,
            rgba(255,0,127,.09),
            transparent 25%
        ),
        radial-gradient(
            circle at 50% 90%,
            rgba(112,0,255,.10),
            transparent 35%
        ),
        #030008;

    color: #ffffff;

    min-height: 100vh;
}

/* ========================================================
   HIDE STREAMLIT BRANDING
======================================================== */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ========================================================
   MAIN CONTAINER
======================================================== */

.block-container {

    max-width: 1250px;

    padding-top: 1rem !important;
    padding-bottom: 5rem !important;

}

/* ========================================================
   ANIMATED PARTICLE-LIKE BACKGROUND
======================================================== */

.stApp::before {

    content: "";

    position: fixed;

    inset: 0;

    pointer-events: none;

    z-index: 0;

    background-image:
        radial-gradient(
            circle,
            rgba(0,243,255,.55) 1px,
            transparent 1px
        ),
        radial-gradient(
            circle,
            rgba(255,0,127,.35) 1px,
            transparent 1px
        );

    background-size:
        85px 85px,
        125px 125px;

    background-position:
        0 0,
        30px 40px;

    opacity: .20;

    animation: particleMove 22s linear infinite;
}

@keyframes particleMove {

    from {
        transform: translate3d(0,0,0);
    }

    to {
        transform: translate3d(-80px,-100px,0);
    }

}

/* ========================================================
   TOP NAV
======================================================== */

.mo-nav {

    position: relative;

    z-index: 5;

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 18px 24px;

    margin-bottom: 20px;

    border:

        1px solid
        rgba(255,255,255,.10);

    border-radius: 20px;

    background:
        rgba(5,2,12,.55);

    backdrop-filter:
        blur(25px);

    box-shadow:

        0 15px 50px
        rgba(0,0,0,.45),

        inset 0 1px 0
        rgba(255,255,255,.08);
}

.mo-logo {

    display: flex;

    align-items: center;

    gap: 12px;
}

.mo-logo-icon {

    width: 46px;

    height: 46px;

    border-radius: 14px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 21px;

    background:
        linear-gradient(
            135deg,
            #00f3ff,
            #7000ff,
            #ff007f
        );

    box-shadow:

        0 0 18px
        rgba(0,243,255,.65),

        0 0 35px
        rgba(255,0,127,.25);

    animation:
        logoPulse 3s ease-in-out infinite alternate;
}

@keyframes logoPulse {

    from {
        transform: scale(1);
    }

    to {
        transform: scale(1.06);
    }

}

.mo-logo-text {

    font-size: 25px;

    font-weight: 900;

    letter-spacing: .5px;

    background:
        linear-gradient(
            90deg,
            #ffffff,
            #00f3ff,
            #ff007f
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.mo-status {

    display: flex;

    align-items: center;

    gap: 8px;

    padding: 7px 16px;

    border-radius: 30px;

    border:
        1px solid
        rgba(0,243,255,.55);

    background:
        rgba(0,243,255,.07);

    color: #00f3ff;

    font-size: 13px;

    box-shadow:
        0 0 20px
        rgba(0,243,255,.12);
}

.mo-status-dot {

    width: 8px;

    height: 8px;

    border-radius: 50%;

    background: #00f3ff;

    box-shadow:
        0 0 12px
        #00f3ff;

    animation:
        statusBlink 1.2s infinite alternate;
}

@keyframes statusBlink {

    from {
        opacity: .35;
    }

    to {
        opacity: 1;
    }

}

/* ========================================================
   HERO
======================================================== */

.mo-hero {

    position: relative;

    z-index: 2;

    text-align: center;

    padding: 25px 10px 10px;
}

.mo-badge {

    display: inline-flex;

    align-items: center;

    gap: 8px;

    padding: 7px 18px;

    margin-bottom: 22px;

    border-radius: 50px;

    border:
        1px solid
        rgba(255,255,255,.12);

    background:
        linear-gradient(
            90deg,
            rgba(112,0,255,.22),
            rgba(255,0,127,.22)
        );

    backdrop-filter:
        blur(15px);

    font-size: 13px;

    color: #ddd;

    animation:
        heroFloat 4s ease-in-out infinite;
}

@keyframes heroFloat {

    0%,100% {
        transform: translateY(0);
    }

    50% {
        transform: translateY(-7px);
    }

}

.mo-hero-title {

    font-size: clamp(38px, 6vw, 72px);

    line-height: 1.05;

    font-weight: 900;

    margin: 0;

    background:
        linear-gradient(
            180deg,
            #ffffff,
            #a5b4fc
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.mo-hero-title span {

    background:
        linear-gradient(
            90deg,
            #00f3ff,
            #7000ff,
            #ff007f
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;

    filter:
        drop-shadow(
            0 0 25px
            rgba(255,0,127,.45)
        );
}

.mo-description {

    max-width: 720px;

    margin: 20px auto 0;

    color: #9297b5;

    font-size: 16px;

    line-height: 1.9;
}

/* ========================================================
   AI CORE
======================================================== */

.mo-core {

    width: 210px;

    height: 210px;

    margin: 35px auto 30px;

    position: relative;

    display: flex;

    align-items: center;

    justify-content: center;
}

.mo-ring-1,
.mo-ring-2,
.mo-ring-3 {

    position: absolute;

    border-radius: 50%;

    pointer-events: none;
}

.mo-ring-1 {

    width: 100%;

    height: 100%;

    border:
        2px dashed
        #00f3ff;

    box-shadow:
        0 0 25px
        rgba(0,243,255,.35);

    animation:
        rotateClockwise 18s linear infinite;
}

.mo-ring-2 {

    width: 118%;

    height: 118%;

    border:
        1px solid
        rgba(255,0,127,.75);

    animation:
        rotateCounter 13s linear infinite;
}

.mo-ring-3 {

    width: 140%;

    height: 140%;

    border:
        1px solid
        rgba(112,0,255,.25);

    animation:
        rotateClockwise 28s linear infinite;
}

@keyframes rotateClockwise {

    from {
        transform: rotate(0deg);
    }

    to {
        transform: rotate(360deg);
    }

}

@keyframes rotateCounter {

    from {
        transform: rotate(360deg);
    }

    to {
        transform: rotate(0deg);
    }

}

.mo-core-image {

    width: 158px;

    height: 158px;

    object-fit: cover;

    border-radius: 50%;

    border:
        3px solid
        rgba(255,255,255,.16);

    box-shadow:

        0 0 30px
        rgba(112,0,255,.75),

        0 0 70px
        rgba(0,243,255,.20);

    z-index: 5;

    transition:
        transform .5s ease,
        box-shadow .5s ease;
}

.mo-core-image:hover {

    transform:
        scale(1.08)
        rotate(3deg);

    box-shadow:

        0 0 45px
        rgba(0,243,255,.8),

        0 0 100px
        rgba(255,0,127,.4);
}

/* ========================================================
   DASHBOARD
======================================================== */

.mo-dashboard {

    position: relative;

    z-index: 4;

    width: 100%;

    margin:
        10px auto 40px;

    border-radius: 28px;

    border:
        1px solid
        rgba(255,255,255,.12);

    background:
        rgba(10,5,20,.62);

    backdrop-filter:
        blur(30px);

    box-shadow:

        0 35px 90px
        rgba(0,0,0,.65),

        inset 0 1px 0
        rgba(255,255,255,.08);

    overflow: hidden;
}

.mo-dashboard-header {

    display: flex;

    justify-content: space-between;

    align-items: center;

    padding: 16px 22px;

    border-bottom:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(255,255,255,.025);
}

.mo-mode {

    display: inline-flex;

    align-items: center;

    gap: 9px;

    padding: 8px 15px;

    border-radius: 12px;

    color: #fff;

    font-size: 13px;

    background:
        linear-gradient(
            90deg,
            rgba(0,243,255,.15),
            rgba(112,0,255,.15)
        );

    border:
        1px solid
        rgba(0,243,255,.28);

    box-shadow:
        0 0 18px
        rgba(0,243,255,.08);
}

.mo-engine {

    color: #7d849d;

    font-size: 12px;
}

/* ========================================================
   CHAT
======================================================== */

.mo-chat {

    padding: 25px;

    min-height: 260px;

    max-height: 520px;

    overflow-y: auto;
}

.mo-welcome {

    display: flex;

    gap: 13px;

    align-items: flex-start;

    margin-bottom: 15px;
}

.mo-avatar {

    min-width: 42px;

    width: 42px;

    height: 42px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 17px;

    background:
        linear-gradient(
            135deg,
            #7000ff,
            #ff007f
        );

    box-shadow:
        0 0 20px
        rgba(112,0,255,.55);
}

.mo-welcome-box {

    padding: 14px 18px;

    border-radius: 18px;

    background:
        rgba(255,255,255,.045);

    border:
        1px solid
        rgba(255,255,255,.09);

    color: #ddd;

    line-height: 1.8;
}

/* ========================================================
   INPUT DECORATION
======================================================== */

.mo-input-label {

    padding:
        0 25px 12px;

    color: #737991;

    font-size: 12px;
}

/* ========================================================
   STREAMLIT CHAT INPUT
======================================================== */

[data-testid="stChatInput"] {

    position: relative;

    z-index: 10;
}

[data-testid="stChatInput"] textarea {

    background:
        rgba(12,8,22,.92) !important;

    border:
        1px solid
        rgba(255,255,255,.13) !important;

    border-radius: 18px !important;

    color: white !important;

    box-shadow:
        0 0 30px
        rgba(0,243,255,.05) !important;

}

[data-testid="stChatInput"] textarea:focus {

    border-color:
        rgba(0,243,255,.65) !important;

    box-shadow:
        0 0 30px
        rgba(0,243,255,.14) !important;
}

/* ========================================================
   STREAMLIT CHAT MESSAGES
======================================================== */

[data-testid="stChatMessage"] {

    position: relative;

    z-index: 5;

    border:
        1px solid
        rgba(255,255,255,.09);

    border-radius: 20px;

    background:
        rgba(12,8,22,.65);

    backdrop-filter:
        blur(18px);

    margin-bottom: 12px;

    padding: 12px;
}

[data-testid="stChatMessageContent"] {

    line-height: 1.8;
}

/* ========================================================
   SIDEBAR
======================================================== */

section[data-testid="stSidebar"] {

    background:
        linear-gradient(
            180deg,
            #09050f,
            #050307
        );

    border-right:
        1px solid
        rgba(255,255,255,.08);
}

.mo-side-title {

    font-size: 23px;

    font-weight: 900;

    background:
        linear-gradient(
            90deg,
            #00f3ff,
            #ff007f
        );

    -webkit-background-clip: text;

    -webkit-text-fill-color: transparent;
}

.mo-side-card {

    padding: 15px;

    border-radius: 15px;

    border:
        1px solid
        rgba(255,255,255,.08);

    background:
        rgba(255,255,255,.025);

    margin-bottom: 12px;
}

/* ========================================================
   BUTTONS
======================================================== */

.stButton > button {

    border-radius: 12px !important;

    border:
        1px solid
        rgba(255,255,255,.12) !important;

    background:
        rgba(255,255,255,.045) !important;

    color: white !important;

    transition:
        all .25s ease !important;
}

.stButton > button:hover {

    border-color:
        rgba(0,243,255,.55) !important;

    box-shadow:
        0 0 20px
        rgba(0,243,255,.13) !important;

    transform:
        translateY(-1px);
}

/* ========================================================
   SCROLLBAR
======================================================== */

::-webkit-scrollbar {

    width: 7px;
}

::-webkit-scrollbar-track {

    background: #050308;
}

::-webkit-scrollbar-thumb {

    background:
        linear-gradient(
            #00f3ff,
            #ff007f
        );

    border-radius: 20px;
}

/* ========================================================
   MOBILE
======================================================== */

@media (max-width: 700px) {

    .mo-nav {

        padding: 13px 15px;
    }

    .mo-logo-text {

        font-size: 20px;
    }

    .mo-status {

        font-size: 10px;

        padding:
            5px 9px;
    }

    .mo-core {

        width: 170px;

        height: 170px;
    }

    .mo-core-image {

        width: 130px;

        height: 130px;
    }

    .mo-dashboard {

        border-radius: 20px;
    }

}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# TOP NAVIGATION
# =========================================================

status_html = (
    """
    <div class="mo-status">
        <span class="mo-status-dot"></span>
        المحرك الخارق نشط
    </div>
    """
    if connection_ok
    else
    """
    <div class="mo-status"
         style="border-color:rgba(255,60,80,.55);
                color:#ff6375;">
        <span class="mo-status-dot"
              style="background:#ff4055;
                     box-shadow:0 0 12px #ff4055;"></span>
        المحرك غير متصل
    </div>
    """
)

st.markdown(
    f"""
    <div class="mo-nav">

        <div class="mo-logo">

            <div class="mo-logo-icon">
                ⚡
            </div>

            <div class="mo-logo-text">
                Mo Dark AI
            </div>

        </div>

        {status_html}

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HERO
# =========================================================

st.markdown(
    """
    <div class="mo-hero">

        <div class="mo-badge">
            ✨
            الجيل القادم من الذكاء الاصطناعي
        </div>

        <h1 class="mo-hero-title">
            مرحباً بك في عالم
            <span>Mo Dark AI</span>
        </h1>

        <div class="mo-description">
            محرك ذكاء اصطناعي متخصص بالبرمجة،
            بناء المشاريع، تحليل الأكواد، تصحيح الأخطاء
            والهندسة البرمجية المتقدمة.
        </div>

        <div class="mo-core">

            <div class="mo-ring-1"></div>
            <div class="mo-ring-2"></div>
            <div class="mo-ring-3"></div>

            <img
                class="mo-core-image"
                src="https://media.giphy.com/media/26tn33aiTi1jkl6H6/giphy.gif"
                alt="Mo Dark AI Core"
            >

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# DASHBOARD HEADER
# =========================================================

st.markdown(
    """
    <div class="mo-dashboard">

        <div class="mo-dashboard-header">

            <div class="mo-mode">
                ⚙️
                النمط الخارق — Dark Core
            </div>

            <div class="mo-engine">
                Qwen Coder Engine
            </div>

        </div>

    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div class="mo-side-title">🖤 Mo Dark AI</div>',
        unsafe_allow_html=True,
    )

    st.caption(
        "Advanced AI Software Engineer"
    )

    st.divider()

    if connection_ok:
        st.success("🟢 النظام متصل")
    else:
        st.error("🔴 النظام غير متصل")

    st.markdown(
        """
        <div class="mo-side-card">

        <b>🧠 قدرات Mo Dark</b>

        <br><br>

        🏗️ بناء المشاريع<br>
        💻 كتابة الكود<br>
        🐛 Debugging<br>
        🔍 Code Review<br>
        📁 Multi-file Projects<br>
        🐍 Python<br>
        ⚡ JavaScript<br>
        🌐 Web Development<br>
        🗄️ SQL<br>
        🎨 HTML / CSS

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ AI Engine")

    st.code(
        MODEL,
        language="text",
    )

    st.caption(
        "Strict Software Engineering Mode"
    )

    st.divider()

    if st.button(
        "🗑️ مسح المحادثة",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()


# =========================================================
# CHAT AREA
# =========================================================

if not st.session_state.messages:

    st.markdown(
        """
        <div class="mo-welcome">

            <div class="mo-avatar">
                🤖
            </div>

            <div class="mo-welcome-box">

                أهلاً بك يا بطل! 👋

                <br>

                أنا <b>Mo Dark AI</b>،
                نواتك الذكية لبناء البرامج والمشاريع.

                <br>

                اكتب فكرتك، الكود، أو الخطأ
                وأنا أساعدك بتحويله إلى حل عملي.

            </div>

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="
            text-align:center;
            color:#6f758d;
            margin-top:18px;
            font-size:13px;
        ">
            جرّب:
            "ابنيلي مشروع Streamlit كامل لإدارة متجر"
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# DISPLAY CHAT
# =========================================================

for message in st.session_state.messages:

    role = message["role"]
    content = message["content"]

    with st.chat_message(
        role,
        avatar="🧑‍💻" if role == "user" else "🤖",
    ):

        st.markdown(content)


# =========================================================
# CHAT INPUT
# =========================================================

prompt = st.chat_input(
    "اسأل Mo Dark AI أي شيء..."
)


# =========================================================
# AI PROCESSING
# =========================================================

if prompt:

    # -----------------------------------------------------
    # USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message(
        "user",
        avatar="🧑‍💻",
    ):

        st.markdown(prompt)


    # -----------------------------------------------------
    # CONNECTION CHECK
    # -----------------------------------------------------

    if not connection_ok:

        with st.chat_message(
            "assistant",
            avatar="🤖",
        ):

            st.error(
                "Mo Dark AI غير متصل حالياً."
            )

            if connection_error:

                st.code(
                    connection_error,
                    language="text",
                )

        st.stop()


    # -----------------------------------------------------
    # BUILD MESSAGES
    # -----------------------------------------------------

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    messages.extend(
        st.session_state.messages
    )


    # -----------------------------------------------------
    # AI RESPONSE
    # -----------------------------------------------------

    with st.chat_message(
        "assistant",
        avatar="🤖",
    ):

        response_placeholder = st.empty()

        try:

            response = client.chat_completion(

                model=MODEL,

                messages=messages,

                max_tokens=8192,

                temperature=0.12,
            )

            answer = response.choices[0].message.content

            if not answer:

                answer = (
                    "ما حصلت على إجابة من النموذج."
                )

            response_placeholder.markdown(
                answer
            )


            # -------------------------------------------------
            # SAVE RESPONSE
            # -------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                }
            )


        except Exception as e:

            response_placeholder.error(
                "⚠️ صار خطأ أثناء تشغيل Mo Dark AI."
            )

            st.code(
                str(e),
                language="text",
            )
