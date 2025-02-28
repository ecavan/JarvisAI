# pages/03_General_Questions.py
import streamlit as st

st.set_page_config(
    page_title="General Questions",
    page_icon="💬",
    layout="wide"
)

st.title("Ask General Questions")
st.markdown("Ask any general question to the LLM without using the RAG context.")

# Access the shared RAG system from session state
if 'rag_system' not in st.session_state:
    st.error("Error: RAG system not initialized. Please return to the home page.")
    st.stop()

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for i, (question, answer) in enumerate(st.session_state.chat_history):
    st.markdown(f"**Question {i+1}:**")
    st.markdown(question)
    st.markdown(f"**Answer {i+1}:**")
    st.markdown(answer)
    st.markdown("---")

# Question input
user_question = st.text_area("Ask a question", height=150, 
                            placeholder="Enter your question here...")

# Direct question to LLM button
if st.button("Ask LLM"):
    if user_question:
        # Create placeholder for streaming output
        response_container = st.container()
        with response_container:
            st.markdown("### Answer")
            response_placeholder = st.empty()
            
            # Prepare prompt for LLM without context
            prompt = f"""Please answer the following question to the best of your ability:

Question: {user_question}

Answer:"""
            
            # Generate response using MLX with streaming output
            response = st.session_state.rag_system.query_with_mlx_stream(
                prompt, 
                response_placeholder, 
                max_tokens=1024
            )
            
            # Add to chat history
            st.session_state.chat_history.append((user_question, response))
            
            # Force a rerun to update the chat history display
            st.rerun()