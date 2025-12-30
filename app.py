# app.py
import streamlit as st
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import io
import re
from typing import List, Dict, Any
import traceback

# Page configuration
st.set_page_config(
    page_title="PDF Chat & Statistics",
    page_icon="📄",
    layout="wide"
)

# Initialize session state variables
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = None
if "pdf_stats" not in st.session_state:
    st.session_state.pdf_stats = None
if "pdf_processed" not in st.session_state:
    st.session_state.pdf_processed = False


def extract_text_from_pdf(pdf_file) -> tuple:
    """
    Extract text from PDF file and return text content with page information.
    
    Args:
        pdf_file: Uploaded PDF file object
        
    Returns:
        tuple: (full_text, page_count, pages_list)
    """
    try:
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        page_count = len(pdf_reader.pages)
        pages_text = []
        full_text = ""
        
        for page_num, page in enumerate(pdf_reader.pages):
            text = page.extract_text()
            pages_text.append({"page_num": page_num + 1, "text": text})
            full_text += text + "\n"
        
        return full_text, page_count, pages_text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None, 0, []


def calculate_pdf_statistics(full_text: str, page_count: int, pages_text: List[Dict]) -> Dict[str, Any]:
    """
    Calculate various statistics about the PDF document.
    
    Args:
        full_text: Complete text from PDF
        page_count: Number of pages
        pages_text: List of page texts
        
    Returns:
        dict: Statistics dictionary
    """
    # Word count
    words = re.findall(r'\b\w+\b', full_text)
    total_words = len(words)
    avg_words_per_page = total_words / page_count if page_count > 0 else 0
    
    # Character count
    total_chars = len(full_text)
    
    # Estimate images/tables (heuristic: look for common indicators)
    image_indicators = full_text.lower().count("image") + full_text.lower().count("figure")
    table_indicators = full_text.lower().count("table")
    
    # Generate simple summary (first 300 characters)
    summary = full_text[:300].strip() + "..." if len(full_text) > 300 else full_text.strip()
    
    stats = {
        "page_count": page_count,
        "total_words": total_words,
        "avg_words_per_page": round(avg_words_per_page, 2),
        "total_characters": total_chars,
        "estimated_images": image_indicators,
        "estimated_tables": table_indicators,
        "summary": summary
    }
    
    return stats


def create_vector_store(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Create vector store from text using embeddings.
    
    Args:
        text: Input text to process
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
        
    Returns:
        FAISS vectorstore
    """
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings using HuggingFace (free)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        return vectorstore
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


def create_conversation_chain(vectorstore, api_key: str = None):
    """
    Create conversational retrieval chain.
    
    Args:
        vectorstore: FAISS vectorstore
        api_key: OpenAI API key (optional)
        
    Returns:
        ConversationalRetrievalChain
    """
    try:
        # Use OpenAI if API key provided, otherwise show error
        if api_key:
            llm = OpenAI(
                temperature=0.7,
                openai_api_key=api_key,
                model_name="gpt-3.5-turbo-instruct"
            )
        else:
            st.warning("⚠️ No OpenAI API key provided. Please enter your API key in the sidebar.")
            return None
        
        # Create memory
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        # Create conversation chain
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
            memory=memory,
            return_source_documents=True
        )
        
        return conversation_chain
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None


def process_pdf(pdf_file, api_key: str = None):
    """
    Main function to process PDF file.
    
    Args:
        pdf_file: Uploaded PDF file
        api_key: OpenAI API key
    """
    with st.spinner("Processing PDF..."):
        # Extract text
        full_text, page_count, pages_text = extract_text_from_pdf(pdf_file)
        
        if full_text is None or not full_text.strip():
            st.error("Could not extract text from PDF. Please ensure the PDF contains readable text.")
            return
        
        # Calculate statistics
        stats = calculate_pdf_statistics(full_text, page_count, pages_text)
        st.session_state.pdf_stats = stats
        
        # Create vector store
        vectorstore = create_vector_store(full_text)
        
        if vectorstore is None:
            st.error("Failed to create vector store.")
            return
        
        st.session_state.vectorstore = vectorstore
        
        # Create conversation chain
        conversation_chain = create_conversation_chain(vectorstore, api_key)
        st.session_state.conversation_chain = conversation_chain
        
        st.session_state.pdf_processed = True
        st.success("✅ PDF processed successfully! You can now chat with your document.")


def reset_app():
    """Reset the application state."""
    st.session_state.chat_history = []
    st.session_state.vectorstore = None
    st.session_state.conversation_chain = None
    st.session_state.pdf_stats = None
    st.session_state.pdf_processed = False


def main():
    """Main application function."""
    
    # Title
    st.title("📄 PDF Chat & Statistics Generator")
    st.markdown("Upload a PDF and chat with it using AI-powered Retrieval-Augmented Generation (RAG)")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Enter your OpenAI API key to enable chat functionality"
        )
        
        st.markdown("---")
        
        # File upload
        st.header("📤 Upload PDF")
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type="pdf",
            help="Upload a PDF file to analyze and chat with"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 Process PDF"):
                reset_app()
                process_pdf(uploaded_file, api_key)
        
        # Reset button
        if st.session_state.pdf_processed:
            st.markdown("---")
            if st.button("🗑️ Reset & Upload New PDF"):
                reset_app()
                st.rerun()
        
        # Statistics display
        if st.session_state.pdf_stats:
            st.markdown("---")
            st.header("📊 PDF Statistics")
            stats = st.session_state.pdf_stats
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Pages", stats["page_count"])
                st.metric("Total Words", f"{stats['total_words']:,}")
            with col2:
                st.metric("Avg Words/Page", stats["avg_words_per_page"])
                st.metric("Characters", f"{stats['total_characters']:,}")
            
            st.metric("Est. Images", stats["estimated_images"])
            st.metric("Est. Tables", stats["estimated_tables"])
            
            st.markdown("**Document Preview:**")
            st.text_area(
                "First 300 characters",
                stats["summary"],
                height=150,
                disabled=True
            )
    
    # Main content area
    if not st.session_state.pdf_processed:
        # Instructions
        st.info("""
        ### 👋 Welcome! Here's how to use this app:
        
        1. **Enter your OpenAI API Key** in the sidebar (required for chat functionality)
        2. **Upload a PDF file** using the file uploader in the sidebar
        3. **Click "Process PDF"** to analyze the document
        4. **View statistics** about your PDF in the sidebar
        5. **Ask questions** about your document in the chat interface below
        
        #### Example Questions:
        - "What is this document about?"
        - "Summarize the main points"
        - "What does page 3 discuss?"
        - "Are there any conclusions?"
        """)
    else:
        # Chat interface
        st.header("💬 Chat with Your PDF")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message["role"] == "user":
                    st.chat_message("user").write(message["content"])
                else:
                    st.chat_message("assistant").write(message["content"])
        
        # Chat input
        user_question = st.chat_input("Ask a question about your PDF...")
        
        if user_question:
            if not st.session_state.conversation_chain:
                st.error("❌ Please enter your OpenAI API key in the sidebar to enable chat.")
            else:
                # Add user message to chat history
                st.session_state.chat_history.append({
                    "role": "user",
                    "content": user_question
                })
                
                # Display user message
                with st.chat_message("user"):
                    st.write(user_question)
                
                # Get response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        try:
                            response = st.session_state.conversation_chain({
                                "question": user_question
                            })
                            answer = response["answer"]
                            
                            # Add assistant message to chat history
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer
                            })
                            
                            st.write(answer)
                            
                        except Exception as e:
                            error_msg = f"Error generating response: {str(e)}"
                            st.error(error_msg)
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": error_msg
                            })
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Built with Streamlit, LangChain, and HuggingFace 🤖"
        "</div>",
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
