import streamlit as st
import requests
import base64
st.set_page_config(page_title="Multimodal RAG Assistant", layout="centered")
st.markdown(
    """
    <style>
        header[data-testid="stHeader"] {
            background: transparent;
        }
        .sticky-header {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 999;
            background-color: #0e1117; /* Matches Streamlit dark theme */
            border-bottom: 1px solid #262730;
            padding: 12px 24px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .sticky-header-title {
            font-size: 1.15rem;
            font-weight: 600;
            color: #ffffff;
            margin: 0;
            letter-spacing: -0.01em;
        }
        .sticky-header-badge {
            font-size: 0.72rem;
            font-weight: 500;
            color: #60a5fa;
            background-color: rgba(96, 165, 250, 0.12);
            border: 1px solid rgba(96, 165, 250, 0.3);
            padding: 2px 8px;
            border-radius: 999px;
            margin-left: 10px;
        }
        .sticky-header-sub {
            font-size: 0.8rem;
            color: #9ca3af;
            margin: 0;
        }
        /* Push chat messages down slightly so the first message isn't hidden behind the fixed header */
        .block-container {
            padding-top: 4.5rem !important;
        }
    </style>

    <div class="sticky-header">
        <div style="display: flex; align-items: center;">
            <span class="sticky-header-title">DoCU Assistant</span>
            <span class="sticky-header-badge">Multimodal RAG</span>
        </div>
        <span class="sticky-header-sub">Text · Tables · Vision</span>
    </div>
    """,
    unsafe_allow_html=True,
)

API_URL = "http://localhost:8000/chat"
VISUAL_KEYWORDS = ("image", "figure", "diagram", "chart", "plot", "picture", "visual", "architecture" "show")

if "messages" not in st.session_state:
    st.session_state.messages = []

def is_visual_query(query:str) -> bool:
    q = query.lower()
    return any(word in q for word in VISUAL_KEYWORDS)
def render_sources(sources: list[dict], show_inline_image:bool=False):
    if not sources:
        return 

    if show_inline_image:
        for idx, src in enumerate(sources, start=1):
            if src.get("raw_image_b64"):
                image_bytes = base64.b64decode(src["raw_image_b64"])
                st.image(image_bytes, caption=f"Figure from {src.get('source', 'document')} (Source {idx})",
                    use_container_width=True)

    with st.expander("View retrieved multimodal context"):
        for idx, src in enumerate(sources, start=1):
            badges = ["Text"]
            if src.get("has_table"):
                badges.append("Table")
            if src.get("has_image"):
                badges.append("Image")

            st.markdown(
                f"**Source {idx}:** `{src.get('source', 'unknown')}`"
                f"(Chunk ${src.get('chunk_index', -1)}) - **Modallities: **{', '.join(badges)}"
                )
            if src.get("raw_table_html"):
                st.markdown(src["raw_table_html"], unsafe_allow_html=True)

            if src.get("raw_image_b64") and not show_inline_image:
                image_bytes = base64.b64decode(src["raw_image_b64"])
                st.image(image_bytes, caption=f"Retrieved Visual from {src.get('source', 'document')}",
                    use_container_width=True)

            if idx < len(sources):
                st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_sources(msg['sources'], show_inline_image=msg.get("show_inline_image", False))

if prompt := st.chat_input("Ask a question aboutt the text, tables or figures ..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with  st.chat_message("user"):
        st.markdown(prompt)

    trigger_inline = is_visual_query(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving multimodal context and generating answer...."):
            try:
                response = requests.post(API_URL, json={"question": prompt}, timeout=200)
                response.raise_for_status()
                data = response.json()
                answer = data.get("answer", "No answer returned.")
                sources = data.get("sources", [])

                st.markdown(answer)
                render_sources(sources, show_inline_image=trigger_inline)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "show_inline_image": trigger_inline
                    })
            except requests.exceptions.RequestException as e:
                st.error(f"Backend error: {e}")

