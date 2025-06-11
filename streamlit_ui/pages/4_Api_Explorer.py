import streamlit as st
import requests
import json
from urllib.parse import urljoin

# --- Page Configuration ---
st.set_page_config(
    page_title="API Explorer",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 API Explorer")
st.markdown("A dynamic tool to interact with any endpoint of the Graph RAG backend.")
st.info("This page reads the backend's OpenAPI schema to generate the UI. If you add new endpoints, just click 'Refresh API Schema' to see them here.")

# --- Configuration ---
BACKEND_URL = "http://localhost:8000"

# --- Session State Initialization ---
if "api_schema" not in st.session_state:
    st.session_state.api_schema = None
if "endpoint_map" not in st.session_state:
    st.session_state.endpoint_map = {}
if "last_response" not in st.session_state:
    st.session_state.last_response = None

# --- Helper Functions ---
def load_api_schema():
    """Fetches the OpenAPI schema from the backend."""
    try:
        schema_url = urljoin(BACKEND_URL, "/openapi.json")
        response = requests.get(schema_url)
        response.raise_for_status()
        st.session_state.api_schema = response.json()

        endpoint_map = {}
        for path, path_item in st.session_state.api_schema.get("paths", {}).items():
            for method, operation in path_item.items():
                if "multipart/form-data" in str(operation):
                    continue
                key = f"[{method.upper()}] {path}"
                endpoint_map[key] = {"path": path, "method": method.upper(), "details": operation}
        st.session_state.endpoint_map = dict(sorted(endpoint_map.items()))
        st.session_state.last_response = None
        st.success("API schema loaded successfully!")
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch API schema: {e}")
        st.session_state.api_schema = None
        st.session_state.endpoint_map = {}

def make_generic_api_request(method, path_template, path_params, query_params, json_body):
    """Constructs and executes a generic API request."""
    try:
        url = urljoin(BACKEND_URL, path_template.format(**path_params))
        headers = {"Content-Type": "application/json", "Accept": "application/json, */*"}
        st.session_state.last_response = requests.request(
            method=method, url=url, params=query_params, json=json_body, headers=headers, timeout=120
        )
    except requests.exceptions.RequestException as e:
        st.session_state.last_response = e

# --- Sidebar for Endpoint Selection ---
with st.sidebar:
    st.header("Controls")
    if st.button("Load / Refresh API Schema", type="primary"):
        load_api_schema()

    if st.session_state.endpoint_map:
        selected_key = st.selectbox(
            "Select an API Endpoint",
            options=list(st.session_state.endpoint_map.keys()),
            index=None,
            placeholder="Choose an endpoint to test"
        )
    else:
        st.warning("Click the button above to load the API schema.")
        selected_key = None

# --- Main Area for Inputs and Outputs ---
if selected_key:
    endpoint_info = st.session_state.endpoint_map[selected_key]
    details = endpoint_info["details"]

    st.header(f"`{endpoint_info['method']}` `{endpoint_info['path']}`")
    if details.get("summary"): st.subheader(details["summary"])
    if details.get("description"): st.markdown(details["description"])
    st.markdown("---")

    with st.form("api_form"):
        st.subheader("Parameters")
        path_params, query_params, body_data = {}, {}, None

        for param in details.get("parameters", []):
            param_name = param["name"]
            param_in = param["in"]
            param_schema = param.get("schema", {})
            if param_in == "path":
                path_params[param_name] = st.text_input(f"**{param_name}** (path parameter)", help=param.get("description"))
            elif param_in == "query":
                if param_schema.get("type") == "integer":
                    query_params[param_name] = st.number_input(f"{param_name} (query)", value=param_schema.get("default", 0), help=param.get("description"))
                elif param_schema.get("type") == "array":
                    val = st.text_input(f"{param_name} (query, comma-separated)", help=f"{param.get('description')} e.g., file1.pdf,file2.txt")
                    if val: query_params[param_name] = [item.strip() for item in val.split(',')]
                else:
                    query_params[param_name] = st.text_input(f"{param_name} (query)", help=param.get("description"))

        if "requestBody" in details:
            st.subheader("Request Body")

            # --- THIS IS THE FIX ---
            body_schema = details["requestBody"]["content"]["application/json"]["schema"]
            display_schema = body_schema

            # Check if this is a reference that needs to be resolved
            if "$ref" in body_schema:
                ref_path = body_schema["$ref"]
                # Path is like "#/components/schemas/VectorSearchRequest"
                # We split by '/' and take the parts after the '#'
                schema_path_parts = ref_path.split('/')[1:]

                # Navigate the full schema dictionary to find the referenced schema
                resolved_schema = st.session_state.api_schema
                for part in schema_path_parts:
                    resolved_schema = resolved_schema.get(part, {})
                display_schema = resolved_schema
            # --- END OF FIX ---

            with st.expander("View required JSON schema"):
                # We now display the potentially resolved schema
                st.json(display_schema)

            body_str = st.text_area("JSON Body", height=250, placeholder="Enter valid JSON here...")
            if body_str:
                try: body_data = json.loads(body_str)
                except json.JSONDecodeError:
                    st.error("Invalid JSON provided in Request Body.")
                    body_data = "ERROR"

        submitted = st.form_submit_button("🚀 Send Request")

    if submitted:
        if body_data == "ERROR":
            st.warning("Correct the invalid JSON before sending the request.")
        else:
            with st.spinner("Sending request to backend..."):
                make_generic_api_request(
                    method=endpoint_info["method"], path_template=endpoint_info["path"],
                    path_params=path_params, query_params=query_params, json_body=body_data
                )
            st.rerun()

    if st.session_state.last_response:
        st.markdown("---")
        st.header("Response")
        response = st.session_state.last_response
        if isinstance(response, requests.Response):
            st.success(f"**Status Code:** `{response.status_code}`")
            with st.expander("Response Headers"): st.json(dict(response.headers))
            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                st.subheader("JSON Response")
                st.json(response.json())
            elif "application/x-zip-compressed" in content_type or "application/octet-stream" in content_type:
                st.subheader("File Download")
                disposition = response.headers.get('Content-Disposition', '')
                filename = "downloaded_file"
                if "filename=" in disposition:
                    filename = disposition.split('filename=')[1].strip('"')
                st.download_button(label=f"📥 Download {filename}", data=response.content, file_name=filename, mime=content_type)
            else:
                st.subheader("Text Response")
                st.text(response.text)
        elif isinstance(response, Exception):
            st.error(f"An exception occurred: {response}")