import streamlit as st
from mlx_rag_system import MLX_RAG_System

st.set_page_config(
    page_title="Jarvis AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Jarvis AI - RAG System")

st.markdown("""
Welcome to Jarvis AI, your personal AI assistant for document and YouTube video analysis.
""")

# Initialize session state for the RAG system (will be shared across pages)
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = MLX_RAG_System()

# Display the current status of the system
st.subheader("System Status")
st.info(f"MLX Model: {'Loaded' if st.session_state.rag_system.model is not None else 'Not Loaded'}")
st.info(f"Documents Processed: {len(st.session_state.rag_system.processed_docs)}")