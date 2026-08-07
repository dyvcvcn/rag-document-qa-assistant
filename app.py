import os
import tempfile
import streamlit as st

# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --------------------------------------------
# 1. Page Configuration & Setup
# --------------------------------------------
st.set_page_config(page_title="RAG Document Q&A Assistant", page_icon="📚", layout="wide")
st.title("📚 RAG-based Document Q&A Assistant")
st.markdown("""
Upload a PDF/notes set and ask questions answered via retrieval-augmented generation over embedded chunks.
**Tech Stack:** Python, LangChain, FAISS (Vector DB), Google Gemini API
""")

# Sidebar for API Key
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input("Enter Google API Key", type="password")
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    st.markdown("Get your free Gemini API key from [Google AI Studio](https://aistudio.google.com/).")

# --------------------------------------------
# 2. Document Processing & RAG Setup
# --------------------------------------------
uploaded_file = st.file_uploader("Upload your PDF document", type="pdf")

@st.cache_resource(show_spinner=False)
def process_document(file_bytes):
    """Processes the PDF, chunks it, and creates a Vector Store."""
    # Save the uploaded file to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(file_bytes)
        temp_file_path = temp_file.name

    try:
        # Load the PDF
        loader = PyPDFLoader(temp_file_path)
        documents = loader.load()

        # Split text into manageable chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, 
            chunk_overlap=150
        )
        chunks = text_splitter.split_documents(documents)

        # Generate embeddings using Google's text embedding model
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        
        # Store embeddings in a FAISS vector database
        vector_store = FAISS.from_documents(chunks, embeddings)
        return vector_store

    finally:
        # Clean up the temp file
        os.remove(temp_file_path)

if uploaded_file and api_key:
    with st.spinner("Processing document and generating embeddings..."):
        # We read the file bytes and process them
        file_bytes = uploaded_file.read()
        vector_db = process_document(file_bytes)
        st.success("Document successfully processed and embedded!")

    # --------------------------------------------
    # 3. Chat Interface & Question Answering
    # --------------------------------------------
    user_query = st.chat_input("Ask a question about your document...")

    if user_query:
        # Display user question in the UI
        st.chat_message("user").write(user_query)

        # Setup the LLM and the Retrieval Chain
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        retriever = vector_db.as_retriever(search_kwargs={"k": 5}) # Retrieve top 5 closest chunks

        # Define the system prompt guiding the LLM
        system_prompt = (
            "You are a helpful assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the user's question. "
            "If you don't know the answer based on the context, say that you don't know. "
            "Keep your answer concise, clear, and relevant to the document.\n\n"
            "Context:\n{context}"
        )

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        # Create the chains
        question_answer_chain = create_stuff_documents_chain(llm, prompt_template)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain) # type: ignore

        # Generate the response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = rag_chain.invoke({"input": user_query})
                answer = response["answer"]
                st.write(answer)
                
                # Optional: Show the retrieved chunks for transparency
                with st.expander("View Source Chunks"):
                    for i, doc in enumerate(response["context"]):
                        st.markdown(f"**Chunk {i+1}:**\n{doc.page_content}")

elif uploaded_file and not api_key:
    st.warning("⚠️ Please enter your Google API Key in the sidebar to proceed.")