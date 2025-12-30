# app.py
import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import nltk
from collections import Counter
import re
import io
from sklearn.feature_extraction.text import TfidfVectorizer

# Download NLTK data
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Page config
st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "page_count" not in st.session_state:
    st.session_state.page_count = 0
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "llm" not in st.session_state:
    st.session_state.llm = None
if "tokenizer" not in st.session_state:
    st.session_state.tokenizer = None


@st.cache_resource
def load_embedder():
    """Load sentence transformer model for embeddings."""
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


@st.cache_resource
def load_llm():
    """Load local LLM for text generation."""
    tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained('distilgpt2')
    gen_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=150, temperature=0.7)
    return gen_pipeline, tokenizer


def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page_count = len(pdf_reader.pages)
        full_text = ""
        
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text + "\n"
        
        return full_text.strip(), page_count
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None, 0


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    text_len = len(text)
    
    while start < text_len:
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    
    return chunks


def create_vector_store(chunks, embedder):
    """Create FAISS vector store from text chunks."""
    if not chunks:
        return None
    
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    embeddings = np.array(embeddings).astype('float32')
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    return index


def retrieve_context(query, vectorstore, chunks, embedder, top_k=3):
    """Retrieve most relevant chunks for a query."""
    if vectorstore is None or not chunks:
        return ""
    
    query_embedding = embedder.encode([query], show_progress_bar=False)
    query_embedding = np.array(query_embedding).astype('float32')
    
    distances, indices = vectorstore.search(query_embedding, top_k)
    
    context_chunks = [chunks[i] for i in indices[0] if i < len(chunks)]
    return "\n\n".join(context_chunks)


def generate_answer(query, context, llm, tokenizer):
    """Generate answer using local LLM with retrieved context."""
    prompt = f"Context: {context[:800]}\n\nQuestion: {query}\n\nAnswer:"
    
    try:
        response = llm(prompt, max_new_tokens=100, num_return_sequences=1, do_sample=True)
        answer = response[0]['generated_text']
        
        # Extract only the generated answer part
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        else:
            answer = answer[len(prompt):].strip()
        
        # Clean up the answer
        answer = answer.split("\n")[0].strip()
        
        return answer if answer else "I couldn't generate a relevant answer based on the context."
    except Exception as e:
        return f"Error generating answer: {str(e)}"


def calculate_statistics(text):
    """Calculate PDF statistics."""
    # Word count
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    
    # Remove stopwords for frequency analysis
    stop_words = set(stopwords.words('english'))
    filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
    
    # Most common words
    word_freq = Counter(filtered_words)
    top_words = word_freq.most_common(10)
    
    # Extractive summary using TF-IDF
    sentences = sent_tokenize(text)
    if len(sentences) > 0:
        try:
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            if len(sentences) >= 3:
                tfidf_matrix = vectorizer.fit_transform(sentences)
                sentence_scores = tfidf_matrix.sum(axis=1).A1
                top_indices = sentence_scores.argsort()[-3:][::-1]
                summary = " ".join([sentences[i] for i in sorted(top_indices)])
            else:
                summary = " ".join(sentences[:3])
        except:
            summary = " ".join(sentences[:3])
    else:
        summary = "No summary available."
    
    return {
        "total_words": total_words,
        "top_words": top_words,
        "summary": summary
    }


def reset_session():
    """Reset all session state variables."""
    st.session_state.messages = []
    st.session_state.pdf_processed = False
    st.session_state.vectorstore = None
    st.session_state.chunks = []
    st.session_state.full_text = ""
    st.session_state.page_count = 0


# Main UI
st.title("📄 PDF RAG Chat Application")
st.markdown("Upload a PDF, chat with its content, and generate statistics - all running locally!")

# Sidebar
with st.sidebar:
    st.header("📤 Upload PDF")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None:
        if st.button("Process PDF", type="primary"):
            with st.spinner("Processing PDF..."):
                # Load models
                if st.session_state.embedder is None:
                    st.session_state.embedder = load_embedder()
                if st.session_state.llm is None:
                    st.session_state.llm, st.session_state.tokenizer = load_llm()
                
                # Extract text
                full_text, page_count = extract_text_from_pdf(uploaded_file)
                
                if full_text:
                    st.session_state.full_text = full_text
                    st.session_state.page_count = page_count
                    
                    # Create chunks and vector store
                    chunks = chunk_text(full_text)
                    st.session_state.chunks = chunks
                    
                    vectorstore = create_vector_store(chunks, st.session_state.embedder)
                    st.session_state.vectorstore = vectorstore
                    
                    st.session_state.pdf_processed = True
                    st.success(f"✅ PDF processed! {page_count} pages, {len(chunks)} chunks created.")
                else:
                    st.error("Failed to extract text from PDF.")
    
    if st.session_state.pdf_processed:
        st.markdown("---")
        st.markdown(f"**Pages:** {st.session_state.page_count}")
        st.markdown(f"**Chunks:** {len(st.session_state.chunks)}")
        
        if st.button("🗑️ Reset"):
            reset_session()
            st.rerun()

# Main content
if not st.session_state.pdf_processed:
    st.info("👈 Please upload and process a PDF file to start chatting.")
else:
    # Tabs for chat and statistics
    tab1, tab2 = st.tabs(["💬 Chat", "📊 Statistics"])
    
    with tab1:
        st.subheader("Chat with Your PDF")
        
        # Display chat messages
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
                    context = retrieve_context(
                        prompt,
                        st.session_state.vectorstore,
                        st.session_state.chunks,
                        st.session_state.embedder
                    )
                    
                    answer = generate_answer(
                        prompt,
                        context,
                        st.session_state.llm,
                        st.session_state.tokenizer
                    )
                    
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
    
    with tab2:
        st.subheader("PDF Statistics")
        
        if st.button("Generate Statistics", type="primary"):
            with st.spinner("Calculating statistics..."):
                stats = calculate_statistics(st.session_state.full_text)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Total Pages", st.session_state.page_count)
                    st.metric("Total Words", f"{stats['total_words']:,}")
                
                with col2:
                    st.metric("Total Chunks", len(st.session_state.chunks))
                    avg_words = stats['total_words'] // st.session_state.page_count if st.session_state.page_count > 0 else 0
                    st.metric("Avg Words/Page", f"{avg_words:,}")
                
                st.markdown("---")
                st.markdown("### 🔤 Top 10 Most Frequent Words")
                for word, count in stats['top_words']:
                    st.text(f"{word}: {count}")
                
                st.markdown("---")
                st.markdown("### 📝 Extractive Summary")
                st.info(stats['summary'])

st.markdown("---")
st.caption("Built with Streamlit, Sentence Transformers, FAISS, and DistilGPT-2 🚀")
