import streamlit as st
import pandas as pd
from streamlit_ui.helpers import api_request
from io import BytesIO

st.set_page_config(page_title="File Management", layout="wide")
st.title("📄 File Management")
st.markdown("Upload, monitor, and manage the documents in your knowledge base.")

# --- File Uploader ---
with st.expander("Upload New Documents", expanded=True):
    uploaded_files = st.file_uploader(
        "Choose PDF, TXT, DOCX, or MD files",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "md"]
    )
    if st.button("Process Uploaded Files", disabled=not uploaded_files):
        files_to_send = [('files', (f.name, BytesIO(f.getvalue()), f.type)) for f in uploaded_files]
        with st.spinner("Submitting files to the backend for processing..."):
            try:
                response = api_request("POST", "/ingest/upload_files/", files=files_to_send)
                st.success(response.json().get("message", "Files submitted successfully."))
                st.info("Processing happens in the background. Refresh the table below to see status updates.")
            except Exception as e:
                st.error(f"An error occurred during file submission: {e}")

# --- File Status Table ---
st.markdown("---")
st.subheader("Knowledge Base Documents")

if 'file_data' not in st.session_state:
    st.session_state.file_data = []

def refresh_file_list():
    try:
        response = api_request("GET", "/ingest/documents/")
        st.session_state.file_data = response.json()
    except Exception as e:
        st.error(f"Failed to fetch file list: {e}")
        st.session_state.file_data = []

if st.button("🔄 Refresh List"):
    refresh_file_list()

# Display the data
if not st.session_state.file_data:
    st.info("No documents found. Upload files to begin.")
else:
    df = pd.DataFrame(st.session_state.file_data)
    df = df[['filename', 'ingestion_status', 'ingested_at', 'filesize', 'chunk_count', 'entities_added', 'relationships_added', 'error_message']]
    st.dataframe(df, use_container_width=True)

    # --- File Deletion ---
    st.markdown("---")
    st.subheader("Manage Documents")
    files_to_delete = st.multiselect(
        "Select documents to delete",
        options=[f['filename'] for f in st.session_state.file_data]
    )
    if st.button("Delete Selected Documents", disabled=not files_to_delete, type="primary"):
        for filename in files_to_delete:
            with st.spinner(f"Deleting '{filename}' and its associated data..."):
                try:
                    api_request("DELETE", f"/ingest/documents/{filename}")
                    st.success(f"Successfully initiated deletion for '{filename}'.")
                except Exception as e:
                    st.error(f"Failed to delete '{filename}': {e}")
        # Refresh the list after deletion
        refresh_file_list()