import streamlit as st
from openai import OpenAI
import os
from dotenv import load_dotenv
import hashlib
import json
from pathlib import Path

# Initialize client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persistence helpers
APP_DIR = Path(__file__).resolve().parent
STATE_PATH = APP_DIR / ".app_state.json"
DEFAULT_STATE = {
    "vector_store_id": None,
    "num_uploaded_files": 0,
    "file_hashes_in_store": [],
    "existing_filenames_in_store": [],
    "chat_history": [],
}

def load_persisted_state() -> dict:
    try:
        if STATE_PATH.exists():
            with STATE_PATH.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return {**DEFAULT_STATE, **data}
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
            "chat_history": st.session_state.get("chat_history", []),
        }
        with STATE_PATH.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- Streamlit UI ---
st.set_page_config(page_title="Chat with Reports", layout="wide")
st.title("📊 Chat with Annual Reports & Transcripts")

# Load persisted state into session
persisted = load_persisted_state()

# Sidebar controls
st.sidebar.header("Controls")
selected_model = st.sidebar.selectbox(
    "Model",
    options=["gpt-4.1-mini", "gpt-4.1", "o4-mini"],
    index=0,
)
clear_chat = st.sidebar.button("🧹 Clear chat", use_container_width=True)

# Sidebar for file upload
st.sidebar.header("Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# Session state for vector store and counts
if "vector_store_id" not in st.session_state:
    st.session_state.vector_store_id = persisted.get("vector_store_id")
if "num_uploaded_files" not in st.session_state:
    st.session_state.num_uploaded_files = persisted.get("num_uploaded_files", 0)
# Track hashes and filenames to prevent duplicates
if "file_hashes_in_store" not in st.session_state:
    st.session_state.file_hashes_in_store = set(persisted.get("file_hashes_in_store", []))
if "existing_filenames_in_store" not in st.session_state:
    st.session_state.existing_filenames_in_store = set(persisted.get("existing_filenames_in_store", []))
# Persisted chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = persisted.get("chat_history", [])

# Clear chat action
if clear_chat:
    st.session_state.chat_history = []
    persist_state()
    st.sidebar.success("Chat cleared.")

# Ensure a vector store exists or reuse existing by name/id
VECTOR_STORE_NAME = "reports_search_store"

def ensure_vector_store_id() -> str:
    # If we have a stored ID, verify it exists
    existing_id = st.session_state.vector_store_id
    if existing_id:
        try:
            client.vector_stores.retrieve(existing_id)
            return existing_id
        except Exception:
            # Stored ID is invalid or deleted; clear it and continue
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
    # Reset caches for new store
    st.session_state.existing_filenames_in_store = set()
    st.session_state.file_hashes_in_store = set()
    persist_state()
    return store.id

# Upload files to a Vector Store (latest Responses API expects vector_store_ids for file_search)
if uploaded_files:
    with st.spinner("Uploading and indexing files..."):
        vector_store_id = ensure_vector_store_id()

        # If we haven't yet fetched filenames present in the vector store, do it once
        if not st.session_state.existing_filenames_in_store:
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

        # Prepare non-duplicate files
        file_streams = []
        new_filenames = []
        skipped_count = 0
        try:
            for file in uploaded_files:
                file_bytes = file.getbuffer()
                sha256 = hashlib.sha256(file_bytes).hexdigest()

                # Skip if already uploaded this session by hash
                if sha256 in st.session_state.file_hashes_in_store:
                    skipped_count += 1
                    st.sidebar.info(f"Skipped (already uploaded this session): {file.name}")
                    continue

                # Skip if filename already exists in current vector store
                if file.name in st.session_state.existing_filenames_in_store:
                    skipped_count += 1
                    st.sidebar.info(f"Skipped (already in vector store): {file.name}")
                    continue

                # Persist and stage for batch upload
                with open(file.name, "wb") as f:
                    f.write(file_bytes)
                file_streams.append(open(file.name, "rb"))
                new_filenames.append(file.name)
                # Mark as seen to prevent duplicates within same batch
                st.session_state.file_hashes_in_store.add(sha256)
                st.session_state.existing_filenames_in_store.add(file.name)

            if file_streams:
                # Batch upload and index
                client.vector_stores.file_batches.upload_and_poll(
                    vector_store_id=vector_store_id,
                    files=file_streams,
                )
                st.session_state.num_uploaded_files += len(file_streams)
                for name in new_filenames:
                    st.sidebar.success(f"Uploaded: {name}")
                persist_state()
            else:
                if skipped_count > 0:
                    st.sidebar.success("No new files to upload. All selected files were already present.")
        finally:
            for fs in file_streams:
                try:
                    fs.close()
                except Exception:
                    pass

# Layout: Chat left, Docs/Status right
left_col, right_col = st.columns([2, 1])

# Chat interface
with left_col:
    if st.session_state.vector_store_id and st.session_state.num_uploaded_files > 0:
        st.subheader("Chat")

        # Recent messages (visible) above the input
        recent = st.session_state.chat_history[-10:] if st.session_state.chat_history else []
        for speaker, msg in recent:
            role = "user" if speaker.lower().startswith("you") else "assistant"
            with st.chat_message(role):
                st.markdown(msg)

        # Collapsible full chat history
        with st.expander("Full chat history", expanded=False):
            for speaker, msg in st.session_state.chat_history:
                role = "user" if speaker.lower().startswith("you") else "assistant"
                with st.chat_message(role):
                    st.markdown(msg)

        # Chat input (always at the bottom)
        user_input = st.chat_input("Ask about your documents")

        if user_input:
            with st.spinner("Thinking..."):
                # Latest Responses API: pass vector_store_ids to the file_search tool
                response = client.responses.create(
                    model=selected_model,
                    input=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "input_text", "text": user_input},
                            ],
                        }
                    ],
                    tools=[
                        {
                            "type": "file_search",
                            "vector_store_ids": [st.session_state.vector_store_id],
                        }
                    ],
                )

                # Robust answer extraction
                answer = getattr(response, "output_text", None)
                if not answer:
                    collected_parts = []
                    for item in getattr(response, "output", []) or []:
                        content_list = getattr(item, "content", None)
                        if isinstance(content_list, list):
                            for c in content_list:
                                text_val = getattr(c, "text", None)
                                if isinstance(text_val, str) and text_val.strip():
                                    collected_parts.append(text_val)
                    answer = "\n".join(collected_parts).strip() if collected_parts else "(No answer text returned)"

                # Store history and persist
                st.session_state.chat_history.append(("You", user_input))
                st.session_state.chat_history.append(("AI", answer))
                persist_state()

                # Immediately rerun so the new messages render above the input
                st.rerun()
    else:
        st.info("⬅️ Upload at least one PDF file to get started.")

# Right column: status, docs, export
with right_col:
    st.subheader("Status")
    if st.session_state.vector_store_id:
        st.text_input("Vector Store ID", st.session_state.vector_store_id, disabled=True)
    st.metric("Indexed files", st.session_state.num_uploaded_files)

    st.subheader("Documents")
    if st.session_state.existing_filenames_in_store:
        with st.expander("Show document list"):
            for name in sorted(st.session_state.existing_filenames_in_store):
                st.markdown(f"- {name}")
    else:
        st.caption("No documents yet.")

    # Export chat
    def _history_to_markdown(history):
        lines = []
        for spk, msg in history:
            prefix = "### You" if spk.lower().startswith("you") else "### AI"
            lines.append(f"{prefix}\n\n{msg}\n")
        return "\n".join(lines)

    chat_md = _history_to_markdown(st.session_state.chat_history)
    st.download_button(
        label="⬇️ Download chat (Markdown)",
        data=chat_md,
        file_name="chat_history.md",
        mime="text/markdown",
        use_container_width=True,
    )
