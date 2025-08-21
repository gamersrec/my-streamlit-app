import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import hashlib
import json
from pathlib import Path
import tempfile

# --- Init ---
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

APP_DIR = Path(__file__).resolve().parent
STATE_PATH = APP_DIR / ".app_state.json"
DEFAULT_STATE = {
    "vector_store_id": None,
    "num_uploaded_files": 0,
    "file_hashes_in_store": [],
    "existing_filenames_in_store": [],
    "chat_history": [],
}

# --- Persistence helpers ---
def load_persisted_state() -> dict:
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                state = DEFAULT_STATE.copy()
                state.update(data)
                return state
        except Exception:
            pass
    return DEFAULT_STATE.copy()

def persist_state():
    try:
        to_save = {
            "vector_store_id": st.session_state.get("vector_store_id"),
            "num_uploaded_files": st.session_state.get("num_uploaded_files", 0),
            "file_hashes_in_store": list(st.session_state.get("file_hashes_in_store", set())),
            "existing_filenames_in_store": list(st.session_state.get("existing_filenames_in_store", set())),
            "chat_history": st.session_state.get("chat_history", [])[-50:],  # keep last 50
        }
        with STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- Streamlit Config ---
st.set_page_config(page_title="📚 Ask Your Documents", layout="wide")

# --- Custom CSS (simple, friendly) ---
st.markdown(
    """
    <style>
    .app-header h1 { margin-bottom: 0.25rem; }
    .subtle { opacity: 0.8; }
    .chat-bubble-user {
        background-color: #DCF8C6;
        color: black;
        padding: 10px 15px;
        border-radius: 16px;
        margin: 6px 0;
        max-width: 80%;
        float: right;
        clear: both;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .chat-bubble-ai {
        background-color: #F1F0F0;
        color: black;
        padding: 10px 15px;
        border-radius: 16px;
        margin: 6px 0;
        max-width: 80%;
        float: left;
        clear: both;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .sidebar-card {
        background: rgba(240,240,240,0.5);
        padding: 1rem;
        border-radius: 0.75rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(0,0,0,0.05);
    }
    .chip {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        border: 1px solid rgba(0,0,0,0.1);
        margin: 4px 6px 0 0;
        font-size: 0.9rem;
        cursor: pointer;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Load persisted state ---
persisted = load_persisted_state()

# --- Session State ---
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = persisted.get("vector_store_id")
if "num_uploaded_files" not in st.session_state:
    st.session_state.num_uploaded_files = persisted.get("num_uploaded_files", 0)
if "file_hashes_in_store" not in st.session_state:
    st.session_state.file_hashes_in_store = set(persisted.get("file_hashes_in_store", []))
if "existing_filenames_in_store" not in st.session_state:
    st.session_state.existing_filenames_in_store = set(persisted.get("existing_filenames_in_store", []))
if "chat_history" not in st.session_state:
    st.session_state.chat_history = persisted.get("chat_history", [])
# --- MODIFICATION START ---
# Add a flag to track if we are processing a query
if "processing" not in st.session_state:
    st.session_state.processing = False
# --- MODIFICATION END ---

# --- Fixed model (hidden from UI) ---
SELECTED_MODEL = "o4-mini"   # 👈 stays fixed; no selector in UI

# --- Header ---
st.markdown("<div class='app-header'>", unsafe_allow_html=True)
st.title("Ask Your Documents")
st.caption("Upload PDFs and ask questions in everyday language. No technical steps needed.")
st.markdown("</div>", unsafe_allow_html=True)

# --- Sidebar (friendly, minimal) ---
with st.sidebar:
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.header("📂 Your Library")
    st.caption("Add PDFs (reports, transcripts, manuals, etc.)")
    uploaded_files = st.file_uploader("Upload PDF files", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.header("🧹 Start over")
    if st.button("Clear chat history", use_container_width=True):
        st.session_state.chat_history = []
        persist_state()
        st.success("Chat history cleared.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.header("ℹ️ Tips")
    st.caption("• Ask things like “Summarize Q4 results” or “What risks are highlighted?”")
    st.caption("• You can export the chat from the main page.")
    st.markdown("</div>", unsafe_allow_html=True)

# --- Vector Store Management (hidden) ---
VECTOR_STORE_NAME = "reports_search_store"

def ensure_vector_store_id() -> str:
    existing_id = st.session_state.vector_store_id
    if existing_id:
        try:
            client.vector_stores.retrieve(existing_id)
            return existing_id
        except Exception:
            st.session_state.vector_store_id = None
            st.session_state.existing_filenames_in_store = set()
            st.session_state.file_hashes_in_store = set()
            persist_state()

    # Try to find by name
    try:
        stores = client.vector_stores.list(limit=100)
        for store in getattr(stores, "data", []) or []:
            if getattr(store, "name", None) == VECTOR_STORE_NAME:
                st.session_state.vector_store_id = store.id
                persist_state()
                return store.id
    except Exception:
        pass

    # Create new if not found
    store = client.vector_stores.create(name=VECTOR_STORE_NAME)
    st.session_state.vector_store_id = store.id
    st.session_state.existing_filenames_in_store = set()
    st.session_state.file_hashes_in_store = set()
    persist_state()
    return store.id

# --- File Upload & Indexing ---
def prefetch_existing_filenames(vector_store_id: str):
    # Only fetch once per session or when empty
    if st.session_state.existing_filenames_in_store:
        return
    try:
        vs_files = client.vector_stores.files.list(vector_store_id=vector_store_id, limit=100)
        for vs_file in getattr(vs_files, "data", []) or []:
            try:
                fmeta = client.files.retrieve(vs_file.file_id)
                if getattr(fmeta, "filename", None):
                    st.session_state.existing_filenames_in_store.add(fmeta.filename)
            except Exception:
                pass
    except Exception:
        pass
    persist_state()

if uploaded_files:
    with st.spinner("Adding files to your library…"):
        vector_store_id = ensure_vector_store_id()
        prefetch_existing_filenames(vector_store_id)

        file_streams, new_filenames, skipped = [], [], 0
        try:
            for file in uploaded_files:
                file_bytes = file.getbuffer()
                sha256 = hashlib.sha256(file_bytes).hexdigest()

                # skip duplicates
                if sha256 in st.session_state.file_hashes_in_store:
                    skipped += 1
                    st.sidebar.info(f"Skipped duplicate: {file.name}")
                    continue
                if file.name in st.session_state.existing_filenames_in_store:
                    skipped += 1
                    st.sidebar.info(f"Already in library: {file.name}")
                    continue

                # temp save and stage
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(file_bytes)
                tmp.close()
                file_streams.append(open(tmp.name, "rb"))
                new_filenames.append(file.name)

                st.session_state.file_hashes_in_store.add(sha256)
                st.session_state.existing_filenames_in_store.add(file.name)

            if file_streams:
                client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id, files=file_streams
                )
                st.session_state.num_uploaded_files += len(file_streams)
                for name in new_filenames:
                    st.sidebar.success(f"Added: {name}")
                persist_state()
            elif skipped > 0:
                st.sidebar.success("All selected files are already in your library.")
        finally:
            for fs in file_streams:
                try: fs.close()
                except Exception: pass

# --- Layout ---
left_col, right_col = st.columns([2.2, 1])

# --- Chat Area ---
with left_col:
    st.subheader("💬 Ask away")

    # Empty state
    if not st.session_state.vector_store_id or st.session_state.num_uploaded_files == 0:
        st.info("Upload one or more PDFs from the sidebar to get started.")

    # Show recent messages (bubbles)
    for speaker, msg in st.session_state.chat_history[-20:]:
        if speaker == "You":
            st.markdown(f"<div class='chat-bubble-user'>{msg}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-ai'>{msg}</div>", unsafe_allow_html=True)

    # --- MODIFICATION START: Updated Chat Logic ---

    # Quick prompts (chips)
    if not st.session_state.processing:
        st.caption("Try one:")
        qp_cols = st.columns(5)
        quick_prompts = [
            "What is the EBITDA margin?",
            "What is the net income?",
            "What is the customer base?",
            "What is the network coverage?",
            "What is the network speed?",
        ]
        clicked_prompt = None
        for i, prompt in enumerate(quick_prompts):
            if qp_cols[i].button(prompt, key=f"qp_{i}"):
                clicked_prompt = prompt
    else:
        clicked_prompt = None

    # Chat input is now controlled by session state
    user_input = st.chat_input(
        "Type your question…",
        disabled=st.session_state.processing
    )
    query = user_input or clicked_prompt

    # Part 1: Handle new user input
    # If the user enters a query and we are not already processing one.
    if query and not st.session_state.processing:
        st.session_state.processing = True
        st.session_state.chat_history.append(("You", query))
        # Rerun to immediately display the user's message and disable the input field
        st.rerun()

    # Part 2: Process the query if the app is in a "processing" state
    # This part runs on the rerun, after the user's message has been added.
    if st.session_state.processing:
        # Check that the last message is from the user to avoid re-processing
        if st.session_state.chat_history and st.session_state.chat_history[-1][0] == "You":
            last_query = st.session_state.chat_history[-1][1]

            with st.chat_message("assistant"):
                ph = st.empty()
                # Show a "thinking" message as a placeholder
                ph.markdown("<div class='chat-bubble-ai'>Thinking... 🧠</div>", unsafe_allow_html=True)
                streamed = ""
                try:
                    with client.responses.stream(
                        model=SELECTED_MODEL,
                        input=[{
                            "role": "user",
                            "content": [{"type": "input_text", "text": last_query}],
                        }],
                        tools=[{
                            "type": "file_search",
                            "vector_store_ids": [st.session_state.vector_store_id],
                        }],
                    ) as stream:
                        for event in stream:
                            if event.type == "response.output_text.delta":
                                streamed += event.delta
                                ph.markdown(f"<div class='chat-bubble-ai'>{streamed}▌</div>", unsafe_allow_html=True)
                        stream.get_final_response()
                except Exception as e:
                    streamed = "Sorry, I couldn't answer that right now. Please try rephrasing or ask another question."

                # Final message without the blinking cursor
                ph.markdown(f"<div class='chat-bubble-ai'>{streamed}</div>", unsafe_allow_html=True)

            st.session_state.chat_history.append(("AI", streamed))
            persist_state()

        # Reset processing state and rerun to re-enable the input
        st.session_state.processing = False
        st.rerun()

    # --- MODIFICATION END ---

# --- Right Column: Library & Export ---
with right_col:
    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.subheader("📊 Library")
    st.metric("Documents added", st.session_state.num_uploaded_files)
    if st.session_state.existing_filenames_in_store:
        with st.expander("Show files", expanded=False):
            for name in sorted(st.session_state.existing_filenames_in_store):
                st.markdown(f"- {name}")
    else:
        st.caption("No documents yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='sidebar-card'>", unsafe_allow_html=True)
    st.subheader("⬇️ Export chat")
    def _history_to_markdown(history):
        return "\n".join(f"### {spk}\n\n{msg}\n" for spk, msg in history)
    chat_md = _history_to_markdown(st.session_state.chat_history)
    st.download_button(
        label="Download chat (Markdown)",
        data=chat_md,
        file_name="chat_history.md",
        mime="text/markdown",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)