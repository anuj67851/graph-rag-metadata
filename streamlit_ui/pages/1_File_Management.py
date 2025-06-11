import time

import streamlit as st
import pandas as pd
from io import BytesIO

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from streamlit_ui.helpers import api_request

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

    st.markdown("---")
    st.subheader("Document Actions")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("##### Download Documents")
        files_to_download = st.multiselect(
            "Select documents to download as a ZIP file",
            options=[f['filename'] for f in st.session_state.file_data],
            key="download_multiselect"
        )

        # We don't show the button until files are selected
        if files_to_download:
            # Prepare the data for the POST request
            payload = {"filenames": files_to_download}
            try:
                # We "prepare" the download by calling the API.
                # The actual download happens when the user clicks the st.download_button
                response = api_request("POST", "/ingest/documents/download/batch", json=payload)

                # Get the filename from the response headers if available, otherwise create one
                zip_filename = f"GraphRAG_Export_{int(time.time())}.zip"

                st.download_button(
                    label=f"Download Selected ({len(files_to_download)}) as ZIP",
                    data=response.content,
                    file_name=zip_filename,
                    mime='application/zip'
                )
            except Exception as e:
                st.error(f"Could not prepare ZIP file for download: {e}")

    with col2:
        # Action: Delete
        st.markdown("##### Delete Documents")
        files_to_delete = st.multiselect(
            "Select documents to permanently delete",
            options=[f['filename'] for f in st.session_state.file_data],
            key="delete_multiselect"
        )
        if files_to_delete:
            if st.button("Delete Selected Documents", type="primary"):
                for filename in files_to_delete:
                    with st.spinner(f"Deleting '{filename}' and all its associated data..."):
                        try:
                            api_request("DELETE", f"/ingest/documents/{filename}")
                            st.success(f"Successfully initiated deletion for '{filename}'.")
                        except Exception as e:
                            st.error(f"Failed to delete '{filename}': {e}")
                # Refresh the list after deletion
                refresh_file_list()
                st.rerun()

# Initial load of file list if it's empty
if not st.session_state.file_data:
    refresh_file_list()