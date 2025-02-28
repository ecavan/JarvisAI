import os
import tempfile
import streamlit as st

st.set_page_config(
    page_title="PDF Analysis",
    page_icon="📄",
    layout="wide"
)

st.title("PDF Analysis")
col1, col2 = st.columns([3, 1])
with col2:
    if st.button("🔄 Reset All Context", type="primary", help="Clear all stored documents and context"):
        success, message = st.session_state.rag_system.reset_system()
        if success:
            st.success(message)
            # Clear all related session states
            for key in list(st.session_state.keys()):
                if key != 'rag_system':  # Keep the RAG system object
                    del st.session_state[key]
            st.rerun()
        else:
            st.error(message)
st.markdown("Upload a PDF document and ask questions about its content.")

# Access the shared RAG system from session state
if 'rag_system' not in st.session_state:
    st.error("Error: RAG system not initialized. Please return to the home page.")
    st.stop()

# File uploader
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    # Create a hash of the file to use as ID
    doc_id = f"{uploaded_file.name}_{hash(uploaded_file.getvalue())}"
    
    # Create a container for processing messages
    process_container = st.container()
    
    # Check if document was already processed
    if doc_id not in st.session_state.rag_system.processed_docs:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name
        
        # Process PDF
        with process_container:
            st.write("Processing PDF...")
            pages_dict = st.session_state.rag_system.process_pdf(temp_path)
            
            # Process for vector storage
            with st.spinner("Creating vector embeddings..."):
                chunks, chunks_metadata = st.session_state.rag_system.process_pages_for_storage(pages_dict)
                st.session_state.rag_system.add_documents(chunks, chunks_metadata, doc_id)
            
            # Store total pages in session state
            st.session_state.total_pages = len(pages_dict)
            st.success(f"Processed {len(pages_dict)} pages from {uploaded_file.name}")
        
        # Clean up temp file
        os.unlink(temp_path)
    else:
        with process_container:
            st.info(f"Document {uploaded_file.name} already processed")
            # Retrieve page count from rag_system if available
            if hasattr(st.session_state.rag_system, 'current_pdf_pages'):
                st.session_state.total_pages = len(st.session_state.rag_system.current_pdf_pages)

# Page selection
if 'total_pages' in st.session_state and st.session_state.total_pages > 0:
    st.subheader("Query Options")
    
    query_option = st.radio(
        "How would you like to query?",
        ["Search entire document", "Search specific pages"]
    )
    
    filter_dict = None
    if query_option == "Search specific pages":
        selected_pages = st.multiselect(
            "Select pages to search within",
            options=list(range(1, st.session_state.total_pages + 1)),
            format_func=lambda x: f"Page {x}"
        )
        if selected_pages:
            filter_dict = {"page": {"$in": selected_pages}}
    
    # Query interface with larger text area
    st.subheader("Ask Questions")
    query = st.text_area("What would you like to know about the document?", height=100)
    
    # Submit button
    if st.button("Submit Question"):
        if query:
            # Create placeholder for streaming output
            response_container = st.container()
            with response_container:
                st.markdown("### Answer")
                response_placeholder = st.empty()
                
                # Query the model with streaming output
                result = st.session_state.rag_system.query_system(
                    query,
                    response_placeholder,
                    filter_dict
                )
                
                # Display source information
                st.markdown("### Sources")
                for source in result["sources"]:
                    if source["source_type"] == "pdf":
                        st.write(f"- Page {source['page']}")
else:
    if 'total_pages' not in st.session_state:
        st.info("Please upload a PDF document to get started.")
    elif st.session_state.total_pages == 0:
        st.warning("The uploaded PDF doesn't contain any extractable text. Please try another document.")