"""RAG OS Streamlit Web UI with Visual Pipeline Builder."""

import os
import json
import time
from datetime import datetime
from typing import Optional
from uuid import uuid4

import streamlit as st
import requests

# Configuration
API_URL = os.getenv("RAG_OS_API_URL", "http://localhost:8000")

# Page config
st.set_page_config(
    page_title="RAG OS",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for pipeline builder
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .step-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        cursor: pointer;
        transition: transform 0.2s;
    }
    .step-card:hover {
        transform: scale(1.02);
    }
    .step-card-ingestion { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .step-card-chunking { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
    .step-card-embedding { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    .step-card-retrieval { background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }
    .step-card-reranking { background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); color: #333; }
    .step-card-prompt { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .step-card-llm { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
    .step-card-post { background: linear-gradient(135deg, #5ee7df 0%, #b490ca 100%); }

    .pipeline-step {
        background: white;
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        position: relative;
    }
    .pipeline-step.selected {
        border-color: #1E88E5;
        box-shadow: 0 0 10px rgba(30, 136, 229, 0.3);
    }
    .step-connector {
        text-align: center;
        color: #999;
        font-size: 1.5rem;
    }
    .config-panel {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Available step types with their configurations
STEP_TYPES = {
    "ingestion": {
        "name": "Ingestion",
        "icon": "📥",
        "description": "Load documents from files or URLs",
        "color": "ingestion",
        "implementations": {
            "text_file": {"name": "Text File", "config": {"file_path": "", "encoding": "utf-8"}},
            "markdown": {"name": "Markdown", "config": {"file_path": ""}},
            "json": {"name": "JSON", "config": {"file_path": "", "content_field": "content"}},
            "csv": {"name": "CSV", "config": {"file_path": "", "content_column": "content"}},
            "url": {"name": "URL Fetcher", "config": {"url": "", "timeout": 30}},
            "web_crawler": {"name": "Web Crawler", "config": {"start_url": "", "max_depth": 2, "max_pages": 10}},
        }
    },
    "chunking": {
        "name": "Chunking",
        "icon": "✂️",
        "description": "Split documents into chunks",
        "color": "chunking",
        "implementations": {
            "fixed_size": {"name": "Fixed Size", "config": {"chunk_size": 512, "overlap": 50}},
            "sentence": {"name": "Sentence-based", "config": {"max_sentences": 5, "overlap_sentences": 1}},
            "token_aware": {"name": "Token-aware", "config": {"max_tokens": 512, "overlap_tokens": 50, "model": "gpt-4"}},
            "recursive": {"name": "Recursive", "config": {"chunk_size": 512, "separators": ["\\n\\n", "\\n", ". ", " "]}},
        }
    },
    "embedding": {
        "name": "Embedding",
        "icon": "🧮",
        "description": "Generate vector embeddings",
        "color": "embedding",
        "implementations": {
            "openai": {"name": "OpenAI", "config": {"model": "text-embedding-3-small", "dimensions": 1536}},
            "cohere": {"name": "Cohere", "config": {"model": "embed-english-v3.0"}},
            "local": {"name": "Local (sentence-transformers)", "config": {"model": "all-MiniLM-L6-v2"}},
        }
    },
    "retrieval": {
        "name": "Retrieval",
        "icon": "🔍",
        "description": "Retrieve relevant documents",
        "color": "retrieval",
        "implementations": {
            "vector": {"name": "Vector Search", "config": {"top_k": 10, "score_threshold": 0.0}},
            "hybrid": {"name": "Hybrid Search", "config": {"top_k": 10, "vector_weight": 0.7, "keyword_weight": 0.3}},
            "mmr": {"name": "MMR (Diversity)", "config": {"top_k": 10, "diversity": 0.3}},
            "multi_query": {"name": "Multi-Query", "config": {"top_k": 10, "num_queries": 3}},
        }
    },
    "reranking": {
        "name": "Reranking",
        "icon": "📊",
        "description": "Rerank retrieved results",
        "color": "reranking",
        "implementations": {
            "cross_encoder": {"name": "Cross-Encoder", "config": {"model": "cross-encoder/ms-marco-MiniLM-L-6-v2", "top_k": 5}},
            "cohere": {"name": "Cohere Rerank", "config": {"model": "rerank-english-v3.0", "top_k": 5}},
            "recency": {"name": "Recency Boost", "config": {"top_k": 5, "recency_weight": 0.2}},
        }
    },
    "prompt_assembly": {
        "name": "Prompt Assembly",
        "icon": "📝",
        "description": "Build the prompt with context",
        "color": "prompt",
        "implementations": {
            "simple": {"name": "Simple", "config": {"max_context_length": 4000}},
            "template": {"name": "Template-based", "config": {"template": "Context:\\n{context}\\n\\nQuestion: {query}\\n\\nAnswer:"}},
            "chat": {"name": "Chat Format", "config": {"system_prompt": "You are a helpful assistant.", "max_context_length": 4000}},
        }
    },
    "llm_execution": {
        "name": "LLM",
        "icon": "🤖",
        "description": "Generate response with LLM",
        "color": "llm",
        "implementations": {
            "openai": {"name": "OpenAI", "config": {"model": "gpt-4", "temperature": 0.7, "max_tokens": 1000}},
            "anthropic": {"name": "Anthropic Claude", "config": {"model": "claude-3-sonnet-20240229", "temperature": 0.7, "max_tokens": 1000}},
            "mock": {"name": "Mock (Testing)", "config": {"response": "This is a mock response."}},
        }
    },
    "post_processing": {
        "name": "Post-Processing",
        "icon": "✨",
        "description": "Process and format output",
        "color": "post",
        "implementations": {
            "citation": {"name": "Citation Extraction", "config": {"citation_format": "[{n}]"}},
            "formatting": {"name": "Response Formatting", "config": {"format": "markdown"}},
            "validation": {"name": "Answer Validation", "config": {"min_length": 10}},
        }
    },
}


def api_request(endpoint: str, method: str = "GET", data: Optional[dict] = None) -> dict:
    """Make an API request."""
    url = f"{API_URL}{endpoint}"
    try:
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=60)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            raise ValueError(f"Unsupported method: {method}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to API server"}
    except requests.exceptions.Timeout:
        return {"error": "Request timed out"}
    except requests.exceptions.HTTPError as e:
        try:
            return {"error": e.response.json().get("detail", str(e))}
        except:
            return {"error": str(e)}
    except Exception as e:
        return {"error": str(e)}


def check_api_health() -> bool:
    """Check if API is healthy."""
    result = api_request("/health")
    return "error" not in result and result.get("status") == "healthy"


# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{int(time.time())}"
if "pipeline_steps" not in st.session_state:
    st.session_state.pipeline_steps = []
if "selected_step_index" not in st.session_state:
    st.session_state.selected_step_index = None
if "pipeline_name" not in st.session_state:
    st.session_state.pipeline_name = "my-pipeline"


def add_step_to_pipeline(step_type: str, implementation: str):
    """Add a step to the current pipeline."""
    step_info = STEP_TYPES[step_type]
    impl_info = step_info["implementations"][implementation]

    step = {
        "id": f"{step_type}_{len(st.session_state.pipeline_steps)}",
        "type": step_type,
        "implementation": implementation,
        "name": impl_info["name"],
        "config": impl_info["config"].copy(),
        "enabled": True,
    }
    st.session_state.pipeline_steps.append(step)
    st.session_state.selected_step_index = len(st.session_state.pipeline_steps) - 1


def remove_step(index: int):
    """Remove a step from the pipeline."""
    if 0 <= index < len(st.session_state.pipeline_steps):
        st.session_state.pipeline_steps.pop(index)
        if st.session_state.selected_step_index == index:
            st.session_state.selected_step_index = None
        elif st.session_state.selected_step_index and st.session_state.selected_step_index > index:
            st.session_state.selected_step_index -= 1


def move_step(index: int, direction: int):
    """Move a step up or down in the pipeline."""
    new_index = index + direction
    if 0 <= new_index < len(st.session_state.pipeline_steps):
        steps = st.session_state.pipeline_steps
        steps[index], steps[new_index] = steps[new_index], steps[index]
        st.session_state.selected_step_index = new_index


def generate_pipeline_spec() -> dict:
    """Generate pipeline specification from current steps."""
    steps = []
    for i, step in enumerate(st.session_state.pipeline_steps):
        step_spec = {
            "step_id": step["id"],
            "step_type": step["type"],
            "step_class": f"rag_os.steps.{step['type']}.{step['implementation']}",
            "config": step["config"],
            "enabled": step["enabled"],
            "dependencies": [st.session_state.pipeline_steps[i-1]["id"]] if i > 0 else [],
        }
        steps.append(step_spec)

    return {
        "name": st.session_state.pipeline_name,
        "version": "1.0.0",
        "steps": steps,
    }


# Sidebar
with st.sidebar:
    st.markdown("## 🔍 RAG OS")
    st.markdown("---")

    # Navigation
    page = st.radio(
        "Navigation",
        ["Pipeline Builder", "Chat", "Pipelines", "Indexes", "Documents", "Settings"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # API Status
    api_healthy = check_api_health()
    if api_healthy:
        st.success("✅ API Connected")
    else:
        st.error("❌ API Disconnected")

    st.markdown("---")
    st.caption(f"Session: {st.session_state.session_id[:8]}...")


# ============================================
# PIPELINE BUILDER PAGE
# ============================================
if page == "Pipeline Builder":
    st.markdown('<p class="main-header">🔧 Visual Pipeline Builder</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Drag and drop steps to create your RAG pipeline</p>', unsafe_allow_html=True)

    # Top bar with pipeline name and actions
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    with col1:
        st.session_state.pipeline_name = st.text_input(
            "Pipeline Name",
            value=st.session_state.pipeline_name,
            label_visibility="collapsed",
            placeholder="Enter pipeline name..."
        )
    with col2:
        if st.button("💾 Save Pipeline", use_container_width=True):
            if st.session_state.pipeline_steps:
                spec = generate_pipeline_spec()
                result = api_request("/pipelines", method="POST", data=spec)
                if "error" not in result:
                    st.success(f"Pipeline '{st.session_state.pipeline_name}' saved!")
                else:
                    st.error(f"Failed to save: {result['error']}")
            else:
                st.warning("Add some steps first!")
    with col3:
        if st.button("🧪 Test Pipeline", use_container_width=True):
            st.session_state.show_test_panel = True
    with col4:
        if st.button("🗑️ Clear All", use_container_width=True):
            st.session_state.pipeline_steps = []
            st.session_state.selected_step_index = None
            st.rerun()

    st.markdown("---")

    # Main layout: Step palette | Pipeline canvas | Config panel
    palette_col, canvas_col, config_col = st.columns([1, 2, 1.5])

    # Step Palette
    with palette_col:
        st.markdown("### 📦 Step Library")
        st.caption("Click to add steps to your pipeline")

        for step_type, info in STEP_TYPES.items():
            with st.expander(f"{info['icon']} {info['name']}", expanded=False):
                st.caption(info['description'])
                for impl_key, impl_info in info["implementations"].items():
                    if st.button(
                        f"+ {impl_info['name']}",
                        key=f"add_{step_type}_{impl_key}",
                        use_container_width=True,
                    ):
                        add_step_to_pipeline(step_type, impl_key)
                        st.rerun()

    # Pipeline Canvas
    with canvas_col:
        st.markdown("### 🔗 Pipeline Flow")

        if not st.session_state.pipeline_steps:
            st.info("👆 Click steps from the library to add them here")
        else:
            for i, step in enumerate(st.session_state.pipeline_steps):
                step_info = STEP_TYPES.get(step["type"], {})

                # Step container
                is_selected = st.session_state.selected_step_index == i

                col1, col2, col3, col4 = st.columns([0.5, 3, 0.5, 0.5])

                with col1:
                    # Move buttons
                    if i > 0:
                        if st.button("⬆️", key=f"up_{i}", help="Move up"):
                            move_step(i, -1)
                            st.rerun()
                    if i < len(st.session_state.pipeline_steps) - 1:
                        if st.button("⬇️", key=f"down_{i}", help="Move down"):
                            move_step(i, 1)
                            st.rerun()

                with col2:
                    # Step card
                    enabled_icon = "✅" if step["enabled"] else "⏸️"
                    border_color = "#1E88E5" if is_selected else "#e0e0e0"

                    step_html = f"""
                    <div style="
                        background: white;
                        border: 3px solid {border_color};
                        border-radius: 10px;
                        padding: 1rem;
                        margin: 0.3rem 0;
                        {'box-shadow: 0 0 15px rgba(30, 136, 229, 0.4);' if is_selected else ''}
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <div>
                                <span style="font-size: 1.2rem;">{step_info.get('icon', '📦')}</span>
                                <strong style="margin-left: 0.5rem;">{step['name']}</strong>
                                <span style="color: #666; font-size: 0.8rem; margin-left: 0.5rem;">({step['type']})</span>
                            </div>
                            <span>{enabled_icon}</span>
                        </div>
                        <div style="color: #888; font-size: 0.8rem; margin-top: 0.3rem;">
                            ID: {step['id']}
                        </div>
                    </div>
                    """
                    st.markdown(step_html, unsafe_allow_html=True)

                    if st.button("⚙️ Configure", key=f"select_{i}", use_container_width=True):
                        st.session_state.selected_step_index = i
                        st.rerun()

                with col3:
                    # Toggle enabled
                    if st.button("⏯️", key=f"toggle_{i}", help="Toggle enabled"):
                        step["enabled"] = not step["enabled"]
                        st.rerun()

                with col4:
                    # Delete button
                    if st.button("🗑️", key=f"del_{i}", help="Remove step"):
                        remove_step(i)
                        st.rerun()

                # Connector arrow (except for last step)
                if i < len(st.session_state.pipeline_steps) - 1:
                    st.markdown("<div class='step-connector'>↓</div>", unsafe_allow_html=True)

    # Configuration Panel
    with config_col:
        st.markdown("### ⚙️ Step Configuration")

        if st.session_state.selected_step_index is not None and st.session_state.selected_step_index < len(st.session_state.pipeline_steps):
            step = st.session_state.pipeline_steps[st.session_state.selected_step_index]
            step_info = STEP_TYPES.get(step["type"], {})

            st.markdown(f"**{step_info.get('icon', '')} {step['name']}**")
            st.caption(f"Type: {step['type']} | ID: {step['id']}")

            st.markdown("---")

            # Step ID
            new_id = st.text_input("Step ID", value=step["id"], key="config_step_id")
            if new_id != step["id"]:
                step["id"] = new_id

            # Enabled toggle
            step["enabled"] = st.checkbox("Enabled", value=step["enabled"], key="config_enabled")

            st.markdown("---")
            st.markdown("**Configuration**")

            # Dynamic config fields based on step type
            config = step["config"]
            for key, value in config.items():
                if isinstance(value, bool):
                    config[key] = st.checkbox(key.replace("_", " ").title(), value=value, key=f"config_{key}")
                elif isinstance(value, int):
                    config[key] = st.number_input(key.replace("_", " ").title(), value=value, key=f"config_{key}")
                elif isinstance(value, float):
                    config[key] = st.number_input(key.replace("_", " ").title(), value=value, format="%.2f", key=f"config_{key}")
                elif isinstance(value, str):
                    if len(value) > 50 or "template" in key.lower() or "prompt" in key.lower():
                        config[key] = st.text_area(key.replace("_", " ").title(), value=value, key=f"config_{key}")
                    else:
                        config[key] = st.text_input(key.replace("_", " ").title(), value=value, key=f"config_{key}")
                elif isinstance(value, list):
                    config[key] = st.text_input(
                        key.replace("_", " ").title(),
                        value=", ".join(value) if value else "",
                        key=f"config_{key}",
                        help="Comma-separated values"
                    ).split(", ") if st.session_state.get(f"config_{key}") else value

            step["config"] = config

        else:
            st.info("👈 Select a step to configure it")

    # Pipeline Preview / Test Panel
    st.markdown("---")

    preview_col, test_col = st.columns(2)

    with preview_col:
        with st.expander("📄 Pipeline JSON Preview", expanded=False):
            if st.session_state.pipeline_steps:
                spec = generate_pipeline_spec()
                st.json(spec)
            else:
                st.info("Add steps to see the pipeline specification")

    with test_col:
        if st.session_state.get("show_test_panel"):
            with st.expander("🧪 Test Pipeline", expanded=True):
                test_query = st.text_input("Test Query", placeholder="Enter a test question...")
                if st.button("Run Test") and test_query:
                    with st.spinner("Running pipeline..."):
                        # First save the pipeline
                        spec = generate_pipeline_spec()
                        save_result = api_request("/pipelines", method="POST", data=spec)

                        if "error" not in save_result:
                            # Then run a query
                            query_result = api_request("/query", method="POST", data={
                                "query": test_query,
                                "pipeline": st.session_state.pipeline_name,
                            })

                            if "error" not in query_result:
                                st.success("Pipeline executed successfully!")
                                st.markdown("**Result:**")
                                st.write(query_result.get("result", "No result"))
                                st.caption(f"Duration: {query_result.get('duration_ms', 0):.0f}ms")
                            else:
                                st.error(f"Query failed: {query_result['error']}")
                        else:
                            st.error(f"Failed to save pipeline: {save_result['error']}")


# ============================================
# CHAT PAGE
# ============================================
elif page == "Chat":
    st.markdown('<p class="main-header">💬 RAG OS Chat</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Ask questions about your documents</p>', unsafe_allow_html=True)

    # Pipeline selection
    col1, col2 = st.columns([3, 1])
    with col1:
        pipelines = api_request("/pipelines")
        if "error" not in pipelines and pipelines:
            selected_pipeline = st.selectbox("Pipeline", pipelines, label_visibility="collapsed")
        else:
            selected_pipeline = "default"
            st.info("No pipelines available. Create one in Pipeline Builder.")

    with col2:
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")

    # Chat messages
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            with st.chat_message("assistant"):
                st.write(msg["content"])
                if msg.get("duration"):
                    st.caption(f"Response time: {msg['duration']:.0f}ms")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = api_request("/query", method="POST", data={
                    "query": prompt,
                    "pipeline": selected_pipeline,
                    "session_id": st.session_state.session_id,
                })

            if "error" in result:
                st.error(f"Error: {result['error']}")
                response_content = f"Sorry, an error occurred: {result['error']}"
                duration = 0
            else:
                response_content = result.get("result", "No response")
                duration = result.get("duration_ms", 0)
                st.write(response_content)
                st.caption(f"Response time: {duration:.0f}ms")

            st.session_state.messages.append({
                "role": "assistant",
                "content": response_content,
                "duration": duration,
            })


# ============================================
# PIPELINES PAGE
# ============================================
elif page == "Pipelines":
    st.markdown('<p class="main-header">📊 Pipeline Management</p>', unsafe_allow_html=True)

    pipelines = api_request("/pipelines")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ New Pipeline", use_container_width=True):
            st.switch_page_hack = "Pipeline Builder"  # Would need custom implementation
            st.info("Go to Pipeline Builder to create a new pipeline")

    if "error" in pipelines:
        st.error(f"Error loading pipelines: {pipelines['error']}")
    elif not pipelines:
        st.info("No pipelines found. Create one in the Pipeline Builder.")
    else:
        for pipeline_name in pipelines:
            info = api_request(f"/pipelines/{pipeline_name}")

            with st.expander(f"📊 {pipeline_name}", expanded=False):
                if "error" not in info:
                    col1, col2, col3 = st.columns(3)
                    col1.metric("Version", info.get("version", "N/A"))
                    col2.metric("Steps", len(info.get("steps", [])))
                    col3.metric("Status", info.get("status", "active"))

                    st.markdown("**Steps:**")
                    for step in info.get("steps", []):
                        st.markdown(f"- `{step.get('id', 'unknown')}` ({step.get('step_class', 'unknown')})")

                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button(f"📋 Clone", key=f"clone_{pipeline_name}"):
                            st.info("Clone functionality coming soon")
                    with col2:
                        if st.button(f"🗑️ Delete", key=f"del_{pipeline_name}"):
                            result = api_request(f"/pipelines/{pipeline_name}", method="DELETE")
                            if "error" not in result:
                                st.success("Pipeline deleted")
                                st.rerun()
                            else:
                                st.error(f"Failed to delete: {result['error']}")


# ============================================
# INDEXES PAGE
# ============================================
elif page == "Indexes":
    st.markdown('<p class="main-header">📁 Index Management</p>', unsafe_allow_html=True)

    indexes = api_request("/indexes")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Create Index", use_container_width=True):
            st.session_state.show_create_index = True

    if "error" in indexes:
        st.error(f"Error: {indexes['error']}")
    elif not indexes:
        st.info("No indexes found. Create one to start indexing documents.")
    else:
        for index in indexes:
            with st.expander(f"📁 {index.get('name', 'Unknown')}", expanded=False):
                col1, col2, col3 = st.columns(3)
                col1.metric("Documents", index.get("document_count", 0))
                col2.metric("Chunks", index.get("chunk_count", 0))
                col3.metric("Status", index.get("status", "active"))

    if st.session_state.get("show_create_index"):
        st.markdown("---")
        st.subheader("Create New Index")
        with st.form("create_index"):
            name = st.text_input("Index Name")
            embedding_model = st.selectbox("Embedding Model", ["text-embedding-3-small", "text-embedding-3-large"])
            chunk_size = st.slider("Chunk Size", 100, 2000, 512)

            if st.form_submit_button("Create"):
                if name:
                    result = api_request("/indexes", method="POST", data={
                        "name": name,
                        "documents": [{"content": "placeholder"}],
                        "config": {"embedding_model": embedding_model, "chunk_size": chunk_size}
                    })
                    if "error" not in result:
                        st.success(f"Index '{name}' created!")
                        st.session_state.show_create_index = False
                        st.rerun()
                    else:
                        st.error(f"Failed: {result['error']}")


# ============================================
# DOCUMENTS PAGE
# ============================================
elif page == "Documents":
    st.markdown('<p class="main-header">📄 Document Ingestion</p>', unsafe_allow_html=True)

    indexes = api_request("/indexes")
    index_names = [idx.get("name") for idx in indexes] if "error" not in indexes and indexes else []

    if not index_names:
        st.warning("Create an index first before uploading documents.")
    else:
        selected_index = st.selectbox("Target Index", index_names)

        st.subheader("Upload Documents")
        uploaded_files = st.file_uploader(
            "Choose files",
            type=["txt", "md", "json", "csv"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Upload & Index"):
            progress = st.progress(0)
            for i, file in enumerate(uploaded_files):
                content = file.read().decode("utf-8", errors="ignore")
                api_request(f"/indexes/{selected_index}/documents", method="POST", data={
                    "content": content,
                    "source": file.name,
                })
                progress.progress((i + 1) / len(uploaded_files))
            st.success(f"Uploaded {len(uploaded_files)} document(s)")

        st.markdown("---")
        st.subheader("Paste Text")
        with st.form("paste_text"):
            text_content = st.text_area("Content", height=200)
            text_source = st.text_input("Source (optional)")
            if st.form_submit_button("Add Document") and text_content:
                result = api_request(f"/indexes/{selected_index}/documents", method="POST", data={
                    "content": text_content,
                    "source": text_source or "pasted_text",
                })
                if "error" not in result:
                    st.success("Document added!")
                else:
                    st.error(f"Failed: {result['error']}")


# ============================================
# SETTINGS PAGE
# ============================================
elif page == "Settings":
    st.markdown('<p class="main-header">⚙️ Settings</p>', unsafe_allow_html=True)

    st.subheader("API Configuration")
    new_api_url = st.text_input("API URL", value=API_URL)
    if st.button("Update API URL"):
        os.environ["RAG_OS_API_URL"] = new_api_url
        st.success("API URL updated. Refresh to apply.")

    st.markdown("---")
    st.subheader("System Health")
    if check_api_health():
        health = api_request("/health")
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", health.get("status", "unknown"))
        col2.metric("Version", health.get("version", "unknown"))
        col3.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
    else:
        st.error("Cannot connect to API server")

    st.markdown("---")
    st.subheader("About")
    st.markdown("""
    **RAG OS** - A fully customizable RAG Operating System

    Features:
    - 🔧 Visual Pipeline Builder
    - 📊 Multiple step types and implementations
    - 🤖 Multiple LLM providers
    - 📁 Document indexing and retrieval
    - 🔒 Security features
    """)


# Footer
st.markdown("---")
st.caption("RAG OS - Visual Pipeline Builder | Built with Streamlit")
