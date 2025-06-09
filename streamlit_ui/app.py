import streamlit as st
import requests
from io import BytesIO
from pyvis.network import Network
import tempfile
import os

# --- Configuration ---
BACKEND_API_URL = "http://localhost:8000/api/v1"

# --- Helper Functions ---
def api_request(method: str, endpoint: str, **kwargs) -> requests.Response:
    """Helper function to make requests to the FastAPI backend."""
    url = f"{BACKEND_API_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=120, **kwargs) # Add a longer timeout for long processes
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API HTTP error occurred: {http_err} - {http_err.response.text}")
        print(f"API HTTP error: {http_err} - {http_err.response.text}")
        raise
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"API Connection error: Could not connect to the backend at {url}. Ensure it's running.")
        print(f"API Connection error: {conn_err}")
        raise
    except requests.exceptions.RequestException as req_err:
        st.error(f"API Request error: {req_err}")
        print(f"API Request error: {req_err}")
        raise

def display_pyvis_graph(graph_data: dict, graph_id: str) -> None:
    """Renders a graph using Pyvis and displays it in Streamlit."""
    if not graph_data or not graph_data.get("nodes"):
        st.info("No graph data to display.")
        return

    net = Network(notebook=True, cdn_resources='remote', height="500px", width="100%", directed=True)

    # Add nodes with color-coding
    for node_info in graph_data.get("nodes", []):
        node_id = node_info["id"]
        node_label = node_info.get("label", node_id)
        node_type = node_info.get("type", "Unknown")
        properties = node_info.get("properties", {})
        title_parts = [f"ID: {node_id}", f"Type: {node_type}"]
        if properties.get("original_mentions"):
            title_parts.append(f"Aliases: {', '.join(properties['original_mentions'][:3])}")

        color_map = {
            "PERSON": "#FFD700", "ORGANIZATION": "#90EE90", "PROJECT": "#FFA07A",
            "LOCATION": "#ADD8E6", "TECHNOLOGY": "#DA70D6"
        }
        color = color_map.get(node_type.upper(), "#97C2FC")

        net.add_node(node_id, label=node_label, title="\n".join(title_parts), color=color, group=node_type)

    # Add edges
    for edge_info in graph_data.get("edges", []):
        net.add_edge(
            edge_info["source"], edge_info["target"],
            label=edge_info.get("label", ""),
            title=f"Type: {edge_info.get('label', '')}"
        )

    net.set_options("""
    var options = {
      "physics": { "solver": "forceAtlas2Based", "forceAtlas2Based": { "gravitationalConstant": -100, "centralGravity": 0.01, "springLength": 100 } },
      "interaction": { "hover": true, "tooltipDelay": 200 },
      "nodes": { "font": { "size": 12 } },
      "edges": { "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } }, "smooth": { "type": "continuous" } }
    }
    """)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix=f"{graph_id}_") as tmp_file:
            net.save_graph(tmp_file.name)
            with open(tmp_file.name, "r", encoding="utf-8") as html_file:
                st.components.v1.html(html_file.read(), height=510, scrolling=False)
        os.unlink(tmp_file.name)
    except Exception as e:
        st.error(f"Error rendering Pyvis graph: {e}")

# --- Streamlit App Layout ---
st.set_page_config(page_title="Graph RAG Application", layout="wide")
st.title("🧠 Graph RAG Application")
st.markdown("Interact with your documents through a Knowledge Graph built from semantic text chunks.")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Define tabs
tab1_ingest, tab2_query, tab3_explore = st.tabs([
    "📄 Document Ingestion", "💬 Query", "🌐 Graph Explorer"
])

# --- Tab 1: Document Ingestion ---
with tab1_ingest:
    st.header("Upload Documents for Knowledge Graph Ingestion")
    st.markdown("Upload PDF, TXT, DOCX, or Markdown files. The system will create semantic chunks, extract information, and build the knowledge base.")
    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True, type=["pdf", "txt", "docx", "md"])

    if uploaded_files:
        if st.button("Process Uploaded Files"):
            files_to_send = [('files', (f.name, BytesIO(f.getvalue()), f.type)) for f in uploaded_files]
            with st.spinner("Processing files... This may take time (semantic chunking requires multiple LLM calls per document)."):
                try:
                    response = api_request("POST", "/ingest/upload_files/", files=files_to_send)
                    results = response.json()
                    st.success("File processing completed by the backend.")
                    for report in results:
                        status = report.get("status", "Unknown")
                        color = "green" if status == "Completed" else "orange" if status == "Skipped" else "red"
                        st.markdown(
                            f"<p style='color:{color};'><b>{report.get('filename')}</b>: {status} - {report.get('message')} "
                            f"(Entities: {report.get('entities_added', 0)}, Rels: {report.get('relationships_added', 0)})</p>",
                            unsafe_allow_html=True
                        )
                except Exception as e:
                    st.error(f"An unexpected error occurred during file submission: {e}")

# --- Tab 2: Query & Chat ---
with tab2_query:
    st.header("Ask Questions to Your Knowledge Base")

    # Display existing chat messages
    for i, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Display context only for assistant messages
            if message["role"] == "assistant":
                if "source_chunks" in message and message["source_chunks"]:
                    with st.expander("Show Retrieved Text Chunks"):
                        for chunk in message["source_chunks"]:
                            st.info(f"**From: {chunk['source_document']} (Score: {chunk['score']:.4f})**\n\n> {chunk['chunk_text'].replace('  ', ' ')}")

                if "subgraph_context" in message and message["subgraph_context"]["nodes"]:
                    with st.expander("Show Retrieved Knowledge Graph"):
                        display_pyvis_graph(message["subgraph_context"], graph_id=f"chat_graph_{i}")

    # Chat input
    if user_query := st.chat_input("What would you like to know?"):
        # Add and display user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking... (Searching chunks, augmenting with graph, generating answer)"):
                try:
                    payload = {"query": user_query}
                    response = api_request("POST", "/query/", json=payload)
                    query_response_data = response.json()

                    assistant_message = {
                        "role": "assistant",
                        "content": query_response_data.get("llm_answer", "Sorry, I couldn't generate a response."),
                        "source_chunks": query_response_data.get("source_chunks", []),
                        "subgraph_context": query_response_data.get("subgraph_context", {})
                    }
                    st.session_state.messages.append(assistant_message)

                    # Rerun to display the new message and its context expanders correctly
                    st.rerun()

                except requests.exceptions.HTTPError as http_err:
                    error_detail = http_err.response.json().get("detail", str(http_err))
                    st.error(f"An API error occurred: {error_detail}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_detail}"})
                    st.rerun() # Add rerun for consistency

                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Unexpected Error: {e}"})
                    st.rerun() # Add rerun for consistency

# --- Tab 3: Graph Explorer ---
with tab3_explore:
    st.header("Explore the Knowledge Graph")
    st.markdown("Visualize a sample of the full graph or the busiest nodes.")
    explore_option = st.selectbox("Choose an exploration option:", ("View Full Graph Sample", "View Top N Busiest Nodes"))

    if explore_option == "View Full Graph Sample":
        st.subheader("Full Graph Sample")
        node_limit = st.slider("Node Limit", 10, 500, 100, 10, key="fs_nodes")
        edge_limit = st.slider("Edge Limit", 10, 1000, 150, 10, key="fs_edges")
        if st.button("Load Full Graph Sample"):
            with st.spinner("Loading graph sample..."):
                try:
                    response = api_request("GET", "/graph/full_sample", params={"node_limit": node_limit, "edge_limit": edge_limit})
                    graph_data = response.json()
                    st.success(f"Graph sample loaded: {len(graph_data.get('nodes',[]))} nodes, {len(graph_data.get('edges',[]))} edges.")
                    display_pyvis_graph(graph_data, graph_id="full_sample_graph")
                except Exception as e:
                    st.error(f"Failed to load graph sample: {e}")

    elif explore_option == "View Top N Busiest Nodes":
        st.subheader("Top N Busiest Nodes")
        top_n = st.slider("Number of Busiest Nodes (Top N)", 1, 30, 5, 1, key="busiest_top_n")
        if st.button("Load Busiest Nodes"):
            with st.spinner(f"Loading top {top_n} busiest nodes and their neighborhood..."):
                try:
                    response = api_request("GET", "/graph/busiest_nodes", params={"top_n": top_n})
                    graph_data = response.json()
                    st.success(f"Busiest nodes subgraph loaded: {len(graph_data.get('nodes',[]))} nodes, {len(graph_data.get('edges',[]))} edges.")
                    display_pyvis_graph(graph_data, graph_id="busiest_nodes_graph")
                except Exception as e:
                    st.error(f"Failed to load busiest nodes: {e}")

# --- Sidebar ---
st.sidebar.info(
    """
    **About this App:**
    A Retrieval Augmented Generation (RAG) system using a Knowledge Graph.
    The backend retrieves relevant text chunks and augments them with structured graph data before generating an answer.
    """
)
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    st.rerun()