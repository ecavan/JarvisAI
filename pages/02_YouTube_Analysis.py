# pages/02_YouTube_Analysis.py
import os
import tempfile
import streamlit as st
from pytubefix import YouTube
from pytubefix.cli import on_progress
import pandas as pd
import whisper
import time

st.set_page_config(
    page_title="YouTube Analysis",
    page_icon="🎥",
    layout="wide"
)

st.title("YouTube Video Analysis")
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
st.markdown("Analyze YouTube videos through transcription and ask questions about the content.")

# Access the shared RAG system from session state
if 'rag_system' not in st.session_state:
    st.error("Error: RAG system not initialized. Please return to the home page.")
    st.stop()

# Initialize session state variables
if 'transcript' not in st.session_state:
    st.session_state.transcript = None
if 'video_info' not in st.session_state:
    st.session_state.video_info = None
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None

# Custom progress callback for streamlit
def streamlit_progress_callback(stream, chunk, bytes_remaining):
    """Display download progress in Streamlit"""
    total_size = stream.filesize
    bytes_downloaded = total_size - bytes_remaining
    percentage = (bytes_downloaded / total_size) * 100
    
    # Update progress bar (if it exists in session state)
    if 'progress_bar' in st.session_state:
        st.session_state.progress_bar.progress(int(percentage) / 100)
    
    # Update progress text (if it exists in session state)
    if 'progress_text' in st.session_state:
        st.session_state.progress_text.text(f"Downloaded {bytes_downloaded/1024/1024:.1f} MB of {total_size/1024/1024:.1f} MB ({percentage:.1f}%)")

# Functions for YouTube processing
def format_duration(seconds):
    """Convert seconds to HH:MM:SS format"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"

def get_video_info(url):
    """Get video information from YouTube URL"""
    try:
        yt = YouTube(url)
        info = {
            "title": yt.title,
            "author": yt.author,
            "length_seconds": yt.length,
            "length_formatted": format_duration(yt.length),
            "thumbnail_url": yt.thumbnail_url,
            "views": yt.views
        }
        return info
    except Exception as e:
        st.error(f"Error retrieving video information: {e}")
        return None

def download_youtube_audio(url, output_path=None):
    """Download audio from YouTube video using pytubefix"""
    try:
        # Create progress elements
        st.session_state.progress_bar = st.progress(0)
        st.session_state.progress_text = st.empty()
        
        # Initialize YouTube object with progress callback
        yt = YouTube(url, on_progress_callback=streamlit_progress_callback)
        
        # Get audio stream
        audio_stream = yt.streams.get_audio_only()
        
        # Download to specified path or current directory
        if output_path:
            file_path = audio_stream.download(output_path=output_path, filename="audio.mp4")
        else:
            file_path = audio_stream.download(filename="audio.mp4")
        
        # Clear progress elements
        st.session_state.progress_bar.empty()
        st.session_state.progress_text.empty()
        
        return file_path
    except Exception as e:
        if 'progress_bar' in st.session_state:
            st.session_state.progress_bar.empty()
        if 'progress_text' in st.session_state:
            st.session_state.progress_text.empty()
        st.error(f"Error downloading audio: {e}")
        return None

def load_whisper_model(model_size="base"):
    """Load the Whisper model with the specified size"""
    if st.session_state.whisper_model is None:
        with st.spinner(f"Loading Whisper {model_size} model... This may take a moment."):
            st.session_state.whisper_model = whisper.load_model(model_size)
    return st.session_state.whisper_model

def transcribe_with_whisper(audio_path, language=None):
    """Transcribe audio file using Whisper"""
    try:
        # Load the model
        model = load_whisper_model()
        
        # Transcribe audio
        with st.spinner("Transcribing audio... This may take a while for longer videos."):
            # Set language if provided, otherwise Whisper will auto-detect
            options = {}
            if language:
                options["language"] = language
                
            result = model.transcribe(audio_path, **options)
            
            # Format the transcript with timestamps
            segments = result.get("segments", [])
            formatted_transcript = ""
            
            for segment in segments:
                start_time = format_duration(int(segment.get("start", 0)))
                text = segment.get("text", "").strip()
                formatted_transcript += f"[{start_time}] {text}\n\n"
            
            return formatted_transcript
    except Exception as e:
        st.error(f"Error transcribing audio: {e}")
        return None

# Create tabs for different modes
tab1, tab2 = st.tabs(["YouTube URL", "Manual Transcript"])

with tab1:
    st.header("Analyze by YouTube URL")
    
    # YouTube URL input
    youtube_url = st.text_input("Enter YouTube URL", key="youtube_url")
    
    # Model selection
    model_size = st.selectbox(
        "Select Whisper model size", 
        ["tiny", "base", "small", "medium", "large"],
        index=1,  # Default to "base" for balance of speed and accuracy
        help="Larger models are more accurate but slower and require more memory"
    )
    
    # Language selection (optional)
    use_language = st.checkbox("Specify language (optional)", 
                             help="By default, Whisper will auto-detect the language")
    language = None
    if use_language:
        language = st.text_input("Language code (e.g., 'en' for English, 'fr' for French)")
    
    # Get video information
    if st.button("Get Video Info"):
        if youtube_url:
            with st.spinner("Fetching video information..."):
                video_info = get_video_info(youtube_url)
                if video_info:
                    st.session_state.video_info = video_info
    
    # Display video information if available
    if st.session_state.video_info:
        st.subheader("Video Information")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.image(st.session_state.video_info["thumbnail_url"], use_column_width=True)
        
        with col2:
            st.write(f"**Title:** {st.session_state.video_info['title']}")
            st.write(f"**Channel:** {st.session_state.video_info['author']}")
            st.write(f"**Duration:** {st.session_state.video_info['length_formatted']}")
            st.write(f"**Views:** {st.session_state.video_info['views']:,}")
        
        # Transcription button
        if st.button("Download & Transcribe"):
            if youtube_url:
                # Reset current model if model size changed
                if st.session_state.whisper_model is not None and model_size != st.session_state.whisper_model.name:
                    st.session_state.whisper_model = None
                
                # Create temp directory
                temp_dir = tempfile.mkdtemp()
                
                # Download audio
                with st.spinner("Downloading audio from YouTube..."):
                    audio_path = download_youtube_audio(youtube_url, temp_dir)
                    if audio_path:
                        st.success("Audio downloaded successfully")
                        
                        # Transcribe audio
                        transcript = transcribe_with_whisper(audio_path, language)
                        if transcript:
                            st.session_state.transcript = transcript
                            
                            # Process transcript for RAG
                            doc_id = f"youtube_{hash(youtube_url)}"
                            with st.spinner("Processing transcript for RAG system..."):
                                st.session_state.rag_system.process_text_for_storage(
                                    transcript, doc_id, chunk_size=500, chunk_overlap=100
                                )
                            st.success("Transcript added to RAG system")
    
    # Display transcript if available
    if st.session_state.transcript:
        st.subheader("Video Transcript")
        st.text_area("Transcript", st.session_state.transcript, height=300, key="transcript_display")
        
        # Download transcript button
        transcript_text = st.session_state.transcript
        st.download_button(
            label="Download Transcript",
            data=transcript_text,
            file_name="transcript.txt",
            mime="text/plain"
        )

with tab2:
    st.header("Manual Transcript Entry")
    st.markdown("If you already have a transcript, you can paste it here.")
    
    manual_transcript = st.text_area("Paste transcript here", height=300)
    transcript_name = st.text_input("Give this transcript a name", placeholder="e.g., Conference Talk 2023")
    
    if st.button("Process Manual Transcript"):
        if manual_transcript and transcript_name:
            doc_id = f"manual_transcript_{hash(transcript_name)}"
            with st.spinner("Processing transcript for RAG system..."):
                st.session_state.rag_system.process_text_for_storage(
                    manual_transcript, doc_id, chunk_size=500, chunk_overlap=100
                )
            st.success(f"Manual transcript '{transcript_name}' added to RAG system")

# Query Interface
st.header("Ask Questions About Processed Videos")
query = st.text_area("What would you like to know about the videos?", height=100)

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
                response_placeholder
            )
            
            # Display source information
            st.markdown("### Sources")
            for source in result["sources"]:
                if source["source_type"] == "youtube":
                    st.write(f"- YouTube Transcript at timestamp {source.get('timestamp', 'unknown')}")
                elif source["source_type"] == "text":
                    st.write(f"- Manual Transcript: {source['doc_id']}")
                else:
                    st.write(f"- {source['source_type']}: {source['doc_id']}")
