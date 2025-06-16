import streamlit as st

st.set_page_config(
    page_title="Graph RAG Home",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 Welcome to the Graph RAG Application!")
st.markdown("---")
st.markdown(
    """
    This application is an advanced Retrieval Augmented Generation (RAG) system 
    that uses a Knowledge Graph to provide insightful answers from your documents.

    **Navigate to the different sections using the sidebar on the left:**

    - **📄 File Management:** Upload new documents, view the status of ingested files, 
      and manage your knowledge base.

    - **💬 Chat with Documents:** Ask questions in natural language and receive answers 
      augmented with context from both text and the knowledge graph.

    - **🌐 Graph Explorer:** Visually explore the connections and relationships that 
      the system has extracted from your documents.
      
    - **🚀 API Explorer:** Explore the endpoints and parameters of the backend API 
      by sending them requests directly.

    ### How it Works
    1.  **Ingestion:** When you upload a document, it's broken down into semantic chunks.
    2.  **Extraction:** An LLM analyzes each chunk to extract key entities (like people, organizations, projects) and their relationships.
    3.  **Storage:** This information is stored in a hybrid system:
        -   **Weaviate (Vector DB):** Stores the text chunks for fast semantic search.
        -   **Neo4j (Graph DB):** Stores the entities and relationships to understand connections.
    4.  **Retrieval:** When you ask a question, the system retrieves relevant text chunks and related graph data.
    5.  **Generation:** A final LLM uses this rich, combined context to generate a comprehensive and accurate answer.
    """
)
st.info("To get started, please select a page from the sidebar.")

# --- Sidebar ---
st.sidebar.success("Select a page above to begin.")