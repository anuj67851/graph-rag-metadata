import streamlit as st

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from streamlit_ui.helpers import api_request, display_pyvis_graph

st.set_page_config(page_title="Graph Explorer", layout="wide")
st.title("🌐 Knowledge Graph Explorer")
st.markdown("Visualize the full knowledge graph or its most connected nodes.")

if "file_list" not in st.session_state:
    st.session_state.file_list = []

def refresh_available_files():
    try:
        response = api_request("GET", "/ingest/documents/")
        st.session_state.file_list = [f['filename'] for f in response.json()]
    except:
        st.session_state.file_list = []

# Initial load of file list
if not st.session_state.file_list:
    refresh_available_files()

# --- Add filter widget ---
st.sidebar.header("Graph Filters")
if st.sidebar.button("Refresh File List"):
    refresh_available_files()

selected_files = st.sidebar.multiselect(
    "Filter graph by document:",
    options=st.session_state.file_list,
    help="Leave empty to explore the full graph."
)

explore_option = st.selectbox(
    "Choose an exploration option:",
    ("View Full Graph Sample", "View Top N Busiest Nodes")
)

if explore_option == "View Full Graph Sample":
    st.subheader("Full Graph Sample")
    col1, col2 = st.columns(2)
    with col1:
        node_limit = st.slider("Node Limit", 10, 500, 100)
    with col2:
        edge_limit = st.slider("Edge Limit", 10, 1000, 150)

    if st.button("Load Full Graph Sample"):
        with st.spinner("Loading graph sample..."):
            try:
                response = api_request("GET", "/graph/full_sample", params={"node_limit": node_limit, "edge_limit": edge_limit, "filenames": selected_files or []})
                display_pyvis_graph(response.json(), "full_sample_graph")
            except Exception as e:
                st.error(f"Failed to load graph sample: {e}")

elif explore_option == "View Top N Busiest Nodes":
    st.subheader("Top N Busiest Nodes")
    top_n = st.slider("Number of Busiest Nodes (Top N)", 1, 50, 5)

    if st.button("Load Busiest Nodes"):
        with st.spinner(f"Loading top {top_n} busiest nodes..."):
            try:
                response = api_request("GET", "/graph/busiest_nodes", params={"top_n": top_n, "filenames": selected_files or []})
                display_pyvis_graph(response.json(), "busiest_nodes_graph")
            except Exception as e:
                st.error(f"Failed to load busiest nodes: {e}")