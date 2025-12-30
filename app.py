# app.py
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import nltk
from collections import Counter
import re
import textstat
import torch

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords

# Configure page
st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

# Hugging Face API token
HF_TOKEN = "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"

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "index" not in st.session_state:
    st.session_state.index = None
if "stats" not in st.session_state:
    st.session_state.stats = {}
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None
if "model" not in st.session_state:
    st.session_state.model = None


@st.cache_resource
def load_embedding_model():
    """Load sentence transformer model for embeddings."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@st.cache_resource
def load_generation_model():
    """Load Hugging Face generative model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base', token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base', token=HF_TOKEN)
    return tokenizer, model


def extract_text_from_pdf(pdf_file):
    """Extract all text from PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text.strip(), len(pdf_reader.pages)
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None, 0


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def create_faiss_index(chunks, embedder):
    """Create FAISS index from text chunks."""
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index


def retrieve_relevant_chunks(query, index, chunks, embedder, top_k=3):
    """Retrieve most relevant chunks for a query."""
    query_embedding = embedder.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = index.search(query_embedding, min(top_k, len(chunks)))
    
    relevant_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return " ".join(relevant_chunks)


def generate_response(query, context, tokenizer, model):
    """Generate response using FLAN-T5 with context."""
    prompt = f"Context: {context}\n\nQuestion: {query}\n\nAnswer:"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_length=200,
            num_beams=4,
            early_stopping=True,
            temperature=0.7
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response


def calculate_statistics(text, page_count):
    """Calculate various statistics about the PDF."""
    # Word count
    words = word_tokenize(text.lower())
    total_words = len([w for w in words if w.isalnum()])
    
    # Sentence count
    sentences = sent_tokenize(text)
    total_sentences = len(sentences)
    
    # Most frequent words (excluding stopwords)
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w.isalnum() and w not in stop_words and len(w) > 2]
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(10)
    
    # Readability score
    try:
        flesch_score = textstat.flesch_reading_ease(text)
        readability = f"{flesch_score:.2f} (Flesch Reading Ease)"
    except:
        readability = "N/A"
    
    return {
        "pages": page_count,
        "words": total_words,
        "sentences": total_sentences,
        "avg_words_per_page": round(total_words / page_count, 2) if page_count > 0 else 0,
        "top_words": top_words,
        "readability": readability
    }


def process_pdf(pdf_file):
    """Main function to process uploaded PDF."""
    with st.spinner("Processing PDF..."):
        # Extract text
        text, page_count = extract_text_from_pdf(pdf_file)
        
        if not text:
            st.error("Could not extract text from PDF.")
            return
        
        st.session_state.pdf_text = text
        
        # Calculate statistics
        stats = calculate_statistics(text, page_count)
        st.session_state.stats = stats
        
        # Load models if not already loaded
        if st.session_state.embedder is None:
            with st.spinner("Loading embedding model..."):
                st.session_state.embedder = load_embedding_model()
        
        if st.session_state.tokenizer is None or st.session_state.model is None:
            with st.spinner("Loading generation model..."):
                st.session_state.tokenizer, st.session_state.model = load_generation_model()
        
        # Create chunks and index
        chunks = chunk_text(text)
        st.session_state.chunks = chunks
        
        index = create_faiss_index(chunks, st.session_state.embedder)
        st.session_state.index = index
        
        st.session_state.pdf_processed = True
        st.success(f"✅ PDF processed successfully! {page_count} pages, {len(chunks)} chunks created.")


def reset_app():
    """Reset application state."""
    st.session_state.messages = []
    st.session_state.pdf_processed = False
    st.session_state.pdf_text = ""
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.stats = {}


# Main UI
st.title("📄 PDF RAG Chat & Statistics")
st.markdown("Upload a PDF to chat with its content and view statistics")

# Sidebar
with st.sidebar:
    st.header("📤 Upload PDF")
    
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            reset_app()
            process_pdf(uploaded_file)
    
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.subheader("📊 PDF Statistics")
        
        stats = st.session_state.stats
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pages", stats["pages"])
            st.metric("Words", f"{stats['words']:,}")
        with col2:
            st.metric("Sentences", f"{stats['sentences']:,}")
            st.metric("Avg Words/Page", stats["avg_words_per_page"])
        
        st.markdown(f"**Readability:** {stats['readability']}")
        
        st.markdown("**Top 10 Words:**")
        for word, count in stats["top_words"]:
            st.text(f"• {word}: {count}")
        
        st.markdown("---")
        if st.button("🗑️ Reset", type="secondary"):
            reset_app()
            st.rerun()

# Main content
if not st.session_state.pdf_processed:
    st.info("👈 Please upload and process a PDF file to start chatting")
    
    st.markdown("### How to use:")
    st.markdown("""
    1. Upload a PDF file using the sidebar
    2. Click 'Process PDF' to analyze the document
    3. View statistics in the sidebar
    4. Ask questions about the PDF content in the chat
    """)
else:
    st.subheader("💬 Chat with PDF")
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about the PDF..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Retrieve relevant context
                    context = retrieve_relevant_chunks(
                        prompt,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.embedder
                    )
                    
                    # Generate response
                    response = generate_response(
                        prompt,
                        context,
                        st.session_state.tokenizer,
                        st.session_state.model
                    )
                    
                    # Handle empty responses
                    if not response.strip():
                        response = "I couldn't generate a relevant answer. Please try rephrasing your question."
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.markdown("---")
st.caption("Built with Streamlit, Sentence Transformers, FAISS, and FLAN-T5")
