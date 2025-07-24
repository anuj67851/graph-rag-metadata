import streamlit as st
import requests
import textwrap
from pyvis.network import Network
import tempfile
import os

# --- Configuration ---
BACKEND_API_URL = "http://localhost:8000/api/v1"

def api_request(method: str, endpoint: str, **kwargs) -> requests.Response:
    """Helper function to make requests to the FastAPI backend."""
    url = f"{BACKEND_API_URL}{endpoint}"
    try:
        response = requests.request(method, url, timeout=120, **kwargs)
        response.raise_for_status()
        return response
    except requests.exceptions.HTTPError as http_err:
        st.error(f"API Error: {http_err.response.status_code} - {http_err.response.json().get('detail', 'No details provided')}")
        raise
    except requests.exceptions.ConnectionError:
        st.error(f"Connection Error: Could not connect to the backend at {url}. Is it running?")
        raise
    except requests.exceptions.RequestException as req_err:
        st.error(f"Request Error: {req_err}")
        raise

def display_pyvis_graph(graph_data: dict, graph_id: str) -> None:
    """Renders a graph using Pyvis and displays it in Streamlit."""
    if not graph_data or not graph_data.get("nodes"):
        st.info("No graph data to display.")
        return

    net = Network(height="700px", width="100%", directed=True, cdn_resources='remote', select_menu=True, filter_menu=True)

    color_map = { "PERSON": "#FFD700", "ORGANIZATION": "#90EE90", "PROJECT": "#FFA07A", "LOCATION": "#ADD8E6", "TECHNOLOGY": "#DA70D6" }

    for node_info in graph_data.get("nodes", []):
        node_id = node_info["id"]
        node_type = node_info.get("type", "Unknown")
        properties = node_info.get("properties", {})

        title_parts = [f"ID: {node_id}", f"Type: {node_type}"]
        if properties.get('original_mentions'):
            title_parts.append(f"Aliases: {', '.join(properties['original_mentions'])}")
        if properties.get('contexts'):
            full_context = f"Contexts: {', '.join(properties['contexts'])}"
            title_parts.append('\n'.join(textwrap.wrap(full_context, width=80)))

        color = color_map.get(node_type.upper(), "#97C2FC")
        net.add_node(node_id, label=node_info.get("label", node_id), title="\n".join(title_parts), color=color, group=node_type)

    for edge_info in graph_data.get("edges", []):
        net.add_edge(edge_info["source"], edge_info["target"], label=edge_info.get("label", ""), title=f"Type: {edge_info.get('label', '')}")

    options_json = """
    {
      "physics": {
        "solver": "barnesHut",
        "barnesHut": {
          "gravitationalConstant": -8000,
          "centralGravity": 0.5,
          "springLength": 75, 
          "springConstant": 0.01,
          "damping": 0.09,
          "avoidOverlap": 0.1
        },
        "minVelocity": 0.75,
        "stabilization": { "iterations": 150 }
      },
      "interaction": {
        "hover": true,
        "dragNodes": true,
        "dragView": true,
        "zoomView": true
      },
      "nodes": {
        "font": { "size": 12 }
      },
      "edges": {
        "font": {
          "size": 10,
          "align": "middle"
        },
        "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
        "smooth": { "type": "continuous" }
      }
    }
    """
    net.set_options(options_json)

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", prefix=f"{graph_id}_") as tmp:
            net.save_graph(tmp.name)
            # Read the generated file as bytes
            html_bytes = tmp.read()
            # Decode the bytes into a string, replacing any invalid characters
            html_string = html_bytes.decode('utf-8', 'replace')
            st.components.v1.html(html_string, height=750, scrolling=True)
        os.unlink(tmp.name)
    except Exception as e:
        st.error(f"Error rendering Pyvis graph: {e}")