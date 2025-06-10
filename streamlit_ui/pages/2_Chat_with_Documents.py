import streamlit as st

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from streamlit_ui.helpers import api_request, display_pyvis_graph

st.set_page_config(page_title="Chat", layout="wide")
st.title("💬 Chat with Your Documents")

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "file_list" not in st.session_state:
    st.session_state.file_list = []

def refresh_available_files():
    try:
        response = api_request("GET", "/ingest/documents/")
        st.session_state.file_list = [f['filename'] for f in response.json()]
    except:
        st.session_state.file_list = []

# --- Sidebar for Filtering ---
with st.sidebar:
    st.header("Query Options")
    if st.button("Refresh File List"):
        refresh_available_files()

    selected_files = st.multiselect(
        "Filter by document:",
        options=st.session_state.file_list,
        help="Leave empty to query all documents."
    )
    if st.button("Clear Chat History", type="primary"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
# Display existing messages
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if "source_chunks" in message and message["source_chunks"]:
                with st.expander("Show Retrieved Context"):
                    for chunk in message["source_chunks"]:
                        st.info(f"**From: {chunk['source_document']} (Score: {chunk.get('score', 0):.4f})**\n\n> {chunk['chunk_text']}")
            if "subgraph_context" in message and message["subgraph_context"]["nodes"]:
                with st.expander("Show Retrieved Knowledge Graph"):
                    display_pyvis_graph(message["subgraph_context"], f"chat_graph_{i}")

# Chat input
if user_query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("Thinking... (This may take a moment)"):
            try:
                payload = {"query": user_query, "filter_filenames": selected_files or None}
                response = api_request("POST", "/query/", json=payload)
                query_response = response.json()

                assistant_message = {
                    "role": "assistant",
                    "content": query_response.get("llm_answer", "Sorry, I couldn't generate a response."),
                    "source_chunks": query_response.get("source_chunks", []),
                    "subgraph_context": query_response.get("subgraph_context", {})
                }
                st.session_state.messages.append(assistant_message)
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
                st.rerun()

# Initial load of file list if empty
if not st.session_state.file_list:
    refresh_available_files()