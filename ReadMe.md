# **RAG System for PDFs and YouTube Videos**  

This is a multi-page **Streamlit** application that implements a **Retrieval-Augmented Generation (RAG)** system using **MLX** for optimized LLM inference on Apple Silicon. It allows users to analyze PDFs and YouTube videos by extracting and querying relevant content.  

---

## **Features**  

### **1. Home Page**  
- Serves as the main entry point for the application.  
- Initializes the shared **RAG system**.  
- Displays system status and navigation options.  

### **2. PDF Analysis Page**  
- Upload and process **PDF documents**.  
- Query documents with **page filtering** to refine search results.  
- View answers **with source references** from the document.  

### **3. YouTube Analysis Page**  
- Process **YouTube videos** via URL.  
- Download and transcribe videos using **OpenAI Whisper**.  
- Extract text from slides in the video using **OCR**.  
- Manually input transcripts for better control over the data.  

---

## **Enhancements & Improvements**  

This project improves upon a standard RAG system with:  

✅ **Modular Design** – Extracted RAG system into a shared module for reuse.  
✅ **Common Query Interface** – Unified querying for PDFs and YouTube transcripts.  
✅ **Manual Transcript Management** – Users can provide their own transcripts.  

---

## **Installation**  

1. **Clone the repository**  
   ```bash
   git clone <your-repo-url>
   cd <your-repo-directory>
   ```

2. **Install dependencies**  
   ```bash
   pip install streamlit pytube openai opencv-python pytesseract pillow langchain langchain_huggingface langchain_chroma pypdf mlx mlx_lm
   ```

3. **Download the MLX model**  
   ```bash
   huggingface-cli download mlx-community/DeepSeek-R1-Distill-Qwen-14B-4bit --local-dir ~/mlx-models
   ```
4. **Run the application**  
   ```bash
   streamlit run Home.py
   ```
