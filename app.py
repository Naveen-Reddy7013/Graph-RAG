import os
import sys
import json
import time
import tempfile
import streamlit as st
from pypdf import PdfReader
from src.database import GraphDatabaseClient
from src.orchestrator import GraphRAGOrchestrator
from src import llm
from run import extract_text_from_pdf, read_chapters
from typing import Tuple

@st.cache_resource(show_spinner="Verifying Groq API Connectivity...")
def check_groq_connectivity() -> Tuple[bool, str]:
    try:
        # Check if Groq client can be initialized and a simple call succeeded
        client = llm.get_groq_client()
        client.chat.completions.create(
            messages=[{"role": "user", "content": "ping"}],
            model=llm.config.GROQ_MODEL,
            max_tokens=1
        )
        return True, ""
    except Exception as e:
        return False, str(e)

@st.dialog("🚨 Groq API Error")
def show_groq_error_dialog(error_msg: str):
    st.markdown("### Groq API is Unavailable")
    st.write("This engine is configured to run **only** if Groq is available. Local mock/simulated execution has been disabled.")
    st.error(f"Error details: {error_msg}")
    st.write("Please verify that your `GROQ_API_KEY` is set correctly in your `.env` file and that you have active internet connectivity.")
    if st.button("Close"):
        st.rerun()


# Set up Streamlit Page Configuration
st.set_page_config(
    page_title="GraphRAG Book Summarizer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS styling for dark mode/modern look
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #a0aec0;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1a202c;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #2d3748;
        text-align: center;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: bold;
        color: #48bb78;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #a0aec0;
    }
</style>
""", unsafe_allow_html=True)

# Main UI Title
st.markdown("<div class='main-header'>🧬 Hierarchical GraphRAG Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Build a Knowledge Graph of your book, cluster into communities, and run grounded queries.</div>", unsafe_allow_html=True)

# --- Sidebar Configuration Panel ---
st.sidebar.markdown("### ⚙️ Engine Settings")

# Verify Groq availability
groq_ok, groq_err = check_groq_connectivity()
if groq_ok:
    st.sidebar.success("🟢 Groq API: Connected")
    mock_llm = False
else:
    st.sidebar.error(f"🔴 Groq API: Offline\n\n{groq_err}")
    mock_llm = False

# Neo4j Database Instance
database_name = st.sidebar.text_input(
    "Target Neo4j Database", 
    value="storygraph",
    help="Enter the database name inside your running Neo4j Desktop."
)

# Data Ingestion Section
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Data Ingestion")
ingest_source = st.sidebar.radio(
    "Choose Story Source:",
    ("Upload custom PDF", "Use sample text chapters"),
    index=1
)

uploaded_file = None
if ingest_source == "Upload custom PDF":
    uploaded_file = st.sidebar.file_uploader("Upload Story PDF", type=["pdf"])

# Trigger Button for Database loading
ingest_clicked = st.sidebar.button("🔨 Clear & Ingest Graph", use_container_width=True)

# --- Ingestion Process Logic ---
if ingest_clicked:
    groq_ok, groq_err = check_groq_connectivity()
    if not groq_ok:
        show_groq_error_dialog(groq_err)
        st.stop()
        
    st.sidebar.info("Connecting to Database...")
    db_client = GraphDatabaseClient(database_name=database_name)
    
    try:
        # Create database in Neo4j if it does not exist
        db_client.create_new_database(database_name)
        db_client.create_constraints()
        
        # Extract book content
        book_content = ""
        if ingest_source == "Upload custom PDF":
            if uploaded_file is not None:
                # Write to temp file to read
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                with st.spinner("Extracting text from PDF pages..."):
                    book_content = extract_text_from_pdf(tmp_path)
                os.unlink(tmp_path)
            else:
                st.sidebar.error("Please upload a PDF file first!")
        else:
            with st.spinner("Reading chapters folder..."):
                book_content = read_chapters("data/chapters")
                
        if book_content.strip():
            # Run Ingestion
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Wiping old graph data...")
            db_client.clear_database()
            progress_bar.progress(20)
            
            status_text.text("Running Map Maker (Extracting Entities & Links)...")
            orchestrator = GraphRAGOrchestrator(db_client)
            orchestrator.ingest_chapters(book_content)
            progress_bar.progress(100)
            status_text.text("")
            
            st.sidebar.success("✅ Graph Ingested & Communities Clustered Successfully!")
        else:
            st.sidebar.error("Book content is empty.")
            
    except Exception as e:
        st.sidebar.error(f"Ingestion failed: {e}")
    finally:
        db_client.close()

# --- Main Query Panel ---

# Establish DB connection to check stats
db_temp = GraphDatabaseClient(database_name=database_name)
nodes_count = 0
edges_count = 0
if db_temp.driver:
    try:
        nodes_count = len(db_temp.get_nodes())
        edges_count = len(db_temp.get_edges())
    except:
        pass
db_temp.close()

# Display active stats
col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
with col_stat1:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{database_name}</div><div class='metric-label'>Active Database</div></div>", unsafe_allow_html=True)
with col_stat2:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{nodes_count}</div><div class='metric-label'>Graph Nodes</div></div>", unsafe_allow_html=True)
with col_stat3:
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{edges_count}</div><div class='metric-label'>Relationship Edges</div></div>", unsafe_allow_html=True)
with col_stat4:
    llm_mode_str = "Groq Cloud (Online)" if groq_ok else "Groq Cloud (Offline)"
    st.markdown(f"<div class='metric-card'><div class='metric-value'>{llm_mode_str}</div><div class='metric-label'>LLM Mode</div></div>", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Question Input Section
query_text = st.text_input(
    "💬 Ask a question about the story:", 
    value="How is Tony Stark connected to Pepper Potts?"
)

col_q1, col_q2, col_q3 = st.columns([2, 2, 1])
with col_q1:
    mode = st.selectbox("Search Strategy Mode", ["local", "global"], index=0, help="Local Search traces path networks; Global Summarization queries community reports.")
with col_q2:
    max_hops = st.slider("Max Search Hops (Local Search only)", min_value=1, max_value=3, value=2)
with col_q3:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    search_clicked = st.button("🔍 Search", use_container_width=True)

# --- Query Processing Logic ---
if search_clicked:
    if not query_text.strip():
        st.warning("Please type a question.")
    elif nodes_count == 0:
        st.error("The targeted database has 0 nodes. Please run 'Clear & Ingest Graph' first in the sidebar!")
    else:
        groq_ok, groq_err = check_groq_connectivity()
        if not groq_ok:
            show_groq_error_dialog(groq_err)
            st.stop()
            
        db_client = GraphDatabaseClient(database_name=database_name)
        
        try:
            with st.spinner("Retrieving paths & generating response..."):
                orchestrator = GraphRAGOrchestrator(db_client)
                start_time = time.time()
                result = orchestrator.query_pipeline(query_text, mode, max_hops)
                end_time = time.time()
                total_duration = end_time - start_time
                
            # Create interactive tabs to show output
            tab_ans, tab_graph, tab_sources, tab_eval, tab_timings = st.tabs([
                "📝 Answer Narrative", 
                "🕸️ Visited Nodes & Edges", 
                "📚 Grounding Context Sources", 
                "🎯 Faithfulness Audit", 
                "⏱️ Execution Timings"
            ])
            
            # Tab 1: Answer
            with tab_ans:
                st.markdown(result["answer"])
                
            # Tab 2: Visited Nodes & Edges
            with tab_graph:
                st.markdown("### Sub-Graph Path Traversed")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.markdown("**Nodes Visited:**")
                    for node in result["graph_context"]["nodes_visited"]:
                        st.markdown(f"- 🏷️ `{node}`")
                with col_g2:
                    st.markdown("**Edges Traversed:**")
                    for edge in result["graph_context"]["edges_traversed"]:
                        st.markdown(f"- 🔗 `{edge}`")
                        
            # Tab 3: Grounding Sources
            with tab_sources:
                st.markdown("### Sources retrieved from Graph Database")
                for src in result["sources"]:
                    st.markdown(f"**ID: `{src['source_id']}`** (Relevance Score: `{src['relevance_score']:.2f}`)")
                    st.info(src["text_chunk"])
                    st.markdown("---")
                    
            # Tab 4: Evaluation
            with tab_eval:
                st.markdown("### Faithfulness Score Assessment")
                f_score = result["evaluation"]["faithfulness_score"]
                factual_gaps = result["evaluation"]["factual_gaps"]
                
                col_e1, col_e2 = st.columns(2)
                with col_e1:
                    st.metric("Faithfulness Rating", f"{f_score * 100:.1f}%")
                with col_e2:
                    st.metric("Factual Gaps Detected", len(factual_gaps))
                    
                if factual_gaps:
                    st.error(f"Factual gaps identified: {factual_gaps}")
                else:
                    st.success("✅ No hallucinations or factual gaps found! The answer is fully grounded in your Graph database.")
                    
            # Tab 5: Timings
            with tab_timings:
                st.markdown("### Process Phase Timings")
                
                # Render timings
                t_lookup = result["metadata"].get("execution_seconds", 0.0)
                st.markdown(f"- **Graph Traversal & Retrieval**: `{t_lookup:.3f} seconds`")
                st.markdown(f"- **Narrative Generation & Synthesis**: `{t_lookup:.3f} seconds`")
                st.markdown(f"- **Total Query Pipeline Duration**: `{total_duration:.3f} seconds`")
                
        except Exception as e:
            st.error(f"Query processing failed: {e}")
        finally:
            db_client.close()
