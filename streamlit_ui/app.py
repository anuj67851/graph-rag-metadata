import streamlit as st
import requests # To make API calls to FastAPI backend
from io import BytesIO # To handle file uploads
from pyvis.network import Network # For graph visualization
import tempfile # For temporarily saving pyvis html
import os

# --- Configuration ---
# Base URL for the FastAPI backend
# Ensure your FastAPI backend is running (e.g., uvicorn app.main:app --reload --port 8000)
BACKEND_API_URL = "http://localhost:8000/api/v1" # Adjust if your backend runs elsewhere or on a different port/prefix

# --- Helper Functions ---
def api_request(method: str, endpoint: str, **kwargs) -> requests.Response:
    """
    Helper function to make requests to the FastAPI backend.
    Handles common request logic and basic error checking.
    """
    url = f"{BACKEND_API_URL}{endpoint}"
    try:
        response = requests.request(method, url, **kwargs)
        response.raise_for_status() # Raises HTTPError for bad responses (4XX or 5XX)
        return response
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API HTTP error occurred: {http_err} - {http_err.response.text}")
        # Log the error for server-side inspection if needed
        print(f"API HTTP error: {http_err} - {http_err.response.text}")
        raise  # Re-raise to stop further processing in the calling function if critical
    except requests.exceptions.ConnectionError as conn_err:
        st.error(f"API Connection error: Could not connect to the backend at {url}. Ensure the backend is running. Details: {conn_err}")
        print(f"API Connection error: {conn_err}")
        raise
    except requests.exceptions.RequestException as req_err:
        st.error(f"API Request error: {req_err}")
        print(f"API Request error: {req_err}")
        raise
    except Exception as e: # Catch-all for other unexpected errors during request
        st.error(f"An unexpected error occurred while making an API request: {e}")
        print(f"Unexpected API request error: {e}")
        raise


def display_pyvis_graph(graph_data: dict, graph_id: str = "pyvis_graph") -> None:
    """
    Renders a graph using Pyvis and displays it in Streamlit.
    graph_data should be a dictionary with "nodes" and "edges" lists,
    matching our common_models.Subgraph structure.
    """
    if not graph_data or not graph_data.get("nodes"):
        st.info("No graph data to display.")
        return

    net = Network(notebook=True, cdn_resources='remote', height="500px", width="100%", directed=True)

    # Add nodes
    for node_info in graph_data.get("nodes", []):
        node_id = node_info.get("id")
        node_label = node_info.get("label", node_id)
        node_type = node_info.get("type", "Unknown")
        properties = node_info.get("properties", {})

        title_parts = [f"ID: {node_id}", f"Type: {node_type}"]
        if properties.get("aliases"):
            title_parts.append(f"Aliases: {', '.join(properties['aliases'][:3])}")
        if properties.get("contexts"):
            title_parts.append(f"Context: {properties['contexts'][0][:100]}...") # Show first context snippet

        # Assign colors based on type (customize as needed)
        color = "#97C2FC" # Default blue
        if "PERSON" in node_type.upper(): color = "#FFD700" # Gold
        elif "ORGANIZATION" in node_type.upper(): color = "#90EE90" # LightGreen
        elif "PROJECT" in node_type.upper(): color = "#FFA07A" # LightSalmon
        elif "LOCATION" in node_type.upper(): color = "#ADD8E6" # LightBlue

        net.add_node(node_id, label=node_label, title="\n".join(title_parts), color=color, group=node_type)

    # Add edges
    for edge_info in graph_data.get("edges", []):
        source_id = edge_info.get("source")
        target_id = edge_info.get("target")
        edge_label = edge_info.get("label", "")
        properties = edge_info.get("properties", {})

        title_parts = [f"Type: {edge_label}"]
        if properties.get("contexts"):
            title_parts.append(f"Context: {properties['contexts'][0][:100]}...")

        net.add_edge(source_id, target_id, title="\n".join(title_parts), label=edge_label)

    # Configure physics for better layout initially, then allow user to toggle
    # net.show_buttons(filter_=['physics']) # Shows physics config buttons
    # solver can be forceAtlas2Based, barnesHut, repulsiveSpringModel
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.3,
          "springLength": 95,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "solver": "barnesHut",
        "timestep": 0.5,
        "stabilization": {
          "enabled": true,
          "iterations": 1000,
          "updateInterval": 100,
          "onlyDynamicEdges": false,
          "fit": true
        }
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 300
      },
      "nodes": {
        "font": { "size": 12 }
      },
      "edges": {
        "font": { "size": 10, "align": "middle" },
        "arrows": {
          "to": { "enabled": true, "scaleFactor": 0.7 }
        },
        "smooth": { "type": "continuous" }
      }
    }
    """)


    # Save to a temporary HTML file and display
    # Using a unique name for each graph render to avoid caching issues if state changes
    temp_file_name = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix=f"{graph_id}_") as tmp_file:
            net.save_graph(tmp_file.name)
            temp_file_name = tmp_file.name

        with open(temp_file_name, "r", encoding="utf-8") as html_file:
            source_code = html_file.read()
            st.components.v1.html(source_code, height=550, scrolling=False)

    except Exception as e:
        st.error(f"Error rendering Pyvis graph: {e}")
    finally:
        if temp_file_name and os.path.exists(temp_file_name):
            os.unlink(temp_file_name) # Clean up the temporary file


# --- Streamlit App Layout ---
st.set_page_config(page_title="Graph RAG Application", layout="wide")
st.title("🧠 Graph RAG Application")
st.markdown("Interact with your documents through a Knowledge Graph.")

# Initialize session state variables if they don't exist
if "messages" not in st.session_state:
    st.session_state.messages = [] # For chat history in Tab 2

# Define tabs
tab1_ingest, tab2_query, tab3_explore = st.tabs([
    "📄 Document Ingestion",
    "💬 Query & Chat",
    "🌐 Graph Explorer"
])


# --- Tab 1: Document Ingestion ---
with tab1_ingest:
    st.header("Upload Documents for Knowledge Graph Ingestion")
    st.markdown(
        "Upload PDF, TXT, DOCX, or Markdown files to extract information, "
        "build the knowledge graph, and enable querying."
    )

    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=["pdf", "txt", "docx", "md"],
        help="You can upload multiple files of supported types."
    )

    if uploaded_files:
        st.write(f"{len(uploaded_files)} file(s) selected:")
        for up_file in uploaded_files:
            st.write(f"- {up_file.name} ({up_file.type})")

        if st.button("Process Uploaded Files", key="process_files_button"):
            if not uploaded_files:
                st.warning("No files selected to process.")
            else:
                files_to_send = []
                for uploaded_file in uploaded_files:
                    # BytesIO is needed to make it compatible with requests' 'files' parameter
                    bytes_io_file = BytesIO(uploaded_file.getvalue())
                    files_to_send.append(('files', (uploaded_file.name, bytes_io_file, uploaded_file.type)))

                with st.spinner("Processing files... This may take some time depending on the number and size of files."):
                    try:
                        # Use the /ingest/upload_files/ endpoint
                        response = api_request("POST", "/ingest/upload_files/", files=files_to_send)

                        results = response.json() # Expects a list of IngestionStatus
                        st.success("File processing initiated by the backend.")

                        for report in results:
                            if report.get("status") == "Completed" or report.get("status") == "Skipped": # Skipped is also a final state
                                st.info(
                                    f"**{report.get('filename')}**: {report.get('status')} - {report.get('message')} "
                                    f"(Entities: {report.get('entities_added', 0)}, Relationships: {report.get('relationships_added', 0)})"
                                )
                            elif report.get("status") == "Failed":
                                st.error(
                                    f"**{report.get('filename')}**: Failed - {report.get('message')}"
                                )
                            else: # e.g. "Accepted" if we were using background tasks
                                st.info(f"**{report.get('filename')}**: {report.get('status')} - {report.get('message')}")

                    except requests.exceptions.HTTPError as http_err:
                        # Error already displayed by api_request helper, but we can add more context
                        st.error(f"Failed to submit files for processing. Please check backend logs.")
                    except Exception as e:
                        st.error(f"An unexpected error occurred during file submission: {e}")
    else:
        st.info("Upload one or more documents to begin.")


# --- Tab 2: Query & Chat ---
with tab2_query:
    st.header("Ask Questions to Your Knowledge Graph")
    st.markdown(
        "Type your questions below. The system will attempt to answer them based on the "
        "information extracted into the knowledge graph."
    )

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                if "llm_context_input_text" in message and message["llm_context_input_text"]:
                    with st.expander("Show Textual Context Used by LLM"):
                        st.text(message["llm_context_input_text"])
                if "subgraph_context" in message and message["subgraph_context"]:
                    with st.expander("Show Interactive Subgraph Context"):
                        # Use a unique key for each graph render
                        graph_key = f"chat_graph_{st.session_state.messages.index(message)}"
                        display_pyvis_graph(message["subgraph_context"], graph_id=graph_key)


    # Chat input
    if user_query := st.chat_input("What would you like to know?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_query})
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get assistant response
        with st.chat_message("assistant"):
            message_placeholder = st.empty() # For streaming-like effect or loading spinner
            full_response_text = ""
            with st.spinner("Thinking..."):
                try:
                    payload = {"query": user_query}
                    response = api_request("POST", "/query/", json=payload)
                    query_api_response = response.json() # QueryResponse model

                    full_response_text = query_api_response.get("llm_answer", "Sorry, I could not generate a response.")
                    subgraph_data = query_api_response.get("subgraph_context")
                    llm_context_str = query_api_response.get("llm_context_input_text")

                    message_placeholder.markdown(full_response_text)

                    # Add assistant response to chat history along with context
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": full_response_text,
                        "subgraph_context": subgraph_data,
                        "llm_context_input_text": llm_context_str
                    })

                    # Re-render to show expanders correctly after adding to session_state
                    # This happens automatically if we modify session state and Streamlit reruns.
                    # However, for dynamic content inside loop (like graph), explicit display needed.

                    # Display expanders for the latest assistant message
                    if llm_context_str:
                        with st.expander("Show Textual Context Used by LLM", expanded=False): # Start collapsed
                            st.text(llm_context_str)
                    if subgraph_data:
                        with st.expander("Show Interactive Subgraph Context", expanded=False): # Start collapsed
                            latest_graph_key = f"chat_graph_latest_{len(st.session_state.messages)}"
                            display_pyvis_graph(subgraph_data, graph_id=latest_graph_key)

                except requests.exceptions.HTTPError as http_err:
                    error_detail = http_err.response.json().get("detail", str(http_err)) if http_err.response else str(http_err)
                    message_placeholder.error(f"Sorry, I encountered an API error: {error_detail}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {error_detail}"})
                except Exception as e:
                    message_placeholder.error(f"An unexpected error occurred: {e}")
                    st.session_state.messages.append({"role": "assistant", "content": f"Unexpected Error: {e}"})


# --- Tab 3: Graph Explorer ---
with tab3_explore:
    st.header("Explore the Knowledge Graph")
    st.markdown("Visualize a sample of the full graph or the busiest nodes.")

    explore_option = st.selectbox(
        "Choose an exploration option:",
        ("View Full Graph Sample", "View Top N Busiest Nodes"),
        key="explore_select"
    )

    if explore_option == "View Full Graph Sample":
        st.subheader("Full Graph Sample")
        node_limit_fs = st.slider("Node Limit (approx.)", min_value=10, max_value=500, value=100, step=10, key="fs_nodes")
        edge_limit_fs = st.slider("Edge Limit", min_value=10, max_value=1000, value=150, step=10, key="fs_edges")

        if st.button("Load Full Graph Sample", key="load_fs_button"):
            with st.spinner("Loading full graph sample..."):
                try:
                    params = {"node_limit": node_limit_fs, "edge_limit": edge_limit_fs}
                    response = api_request("GET", "/graph/full_sample", params=params)
                    graph_sample_data = response.json()
                    if graph_sample_data and graph_sample_data.get("nodes"):
                        st.success(f"Graph sample loaded: {len(graph_sample_data['nodes'])} nodes, {len(graph_sample_data.get('edges',[]))} edges.")
                        display_pyvis_graph(graph_sample_data, graph_id="full_sample_graph")
                    else:
                        st.info("No data returned for the full graph sample, or the graph is empty.")
                except Exception as e:
                    st.error(f"Failed to load full graph sample: {e}")

    elif explore_option == "View Top N Busiest Nodes":
        st.subheader("Top N Busiest Nodes")
        top_n_busiest = st.slider("Number of Busiest Nodes (Top N)", min_value=1, max_value=30, value=5, step=1, key="busiest_top_n")

        if st.button("Load Busiest Nodes", key="load_busiest_button"):
            with st.spinner(f"Loading top {top_n_busiest} busiest nodes and their neighborhood..."):
                try:
                    params = {"top_n": top_n_busiest}
                    response = api_request("GET", "/graph/busiest_nodes", params=params)
                    busiest_nodes_data = response.json()
                    if busiest_nodes_data and busiest_nodes_data.get("nodes"):
                        st.success(f"Busiest nodes subgraph loaded: {len(busiest_nodes_data['nodes'])} nodes, {len(busiest_nodes_data.get('edges',[]))} edges.")
                        display_pyvis_graph(busiest_nodes_data, graph_id="busiest_nodes_graph")
                    else:
                        st.info("No data returned for busiest nodes, or the graph is empty.")
                except Exception as e:
                    st.error(f"Failed to load busiest nodes: {e}")

st.sidebar.info(
    """
    **About this App:**
    This application demonstrates a Graph RAG (Retrieval Augmented Generation)
    system. You can ingest documents to build a knowledge graph,
    then query this graph using natural language.
    """
)
st.sidebar.markdown("---")
if st.sidebar.button("Clear Chat History", key="clear_chat"):
    st.session_state.messages = []
    st.rerun() # Rerun to reflect the cleared chat