from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import time



prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Question: {input}

    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model='all-minilm')
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.splitted_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.splitted_docs, st.session_state.embeddings)


# Streamlit Titile
st.title("RAG Document Q&A")
# Sidebar -> Settings
st.sidebar.title("Settings")
# Dropdown to select different types of Ollama LLM Models
llm_model = st.sidebar.selectbox(
    "Select Ollama LLM Model",
    [
        "gemma2:2b", "gemma3:1b", "gemma3", "gemma3:12b", "gemma3:27b",
        "deepseek-r1", "deepseek-r1:671b",
        "llama4:scout", "llama4:maverick",
        "llama3.3", "llama3.2", "llama3.2:1b", "llama3.2-vision", "llama3.2-vision:90b",
        "llama3.1", "llama3.1:405b",
        "qwq", "phi4", "phi4-mini",
        "mistral", "moondream", "neural-chat", "starling-lm", "codellama",
        "llama2-uncensored", "llava", "granite3.3"
    ],
)

# Load Ollama LLM
llm = OllamaLLM(model=llm_model)

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is Ready")



user_input = st.text_input("Enter your query for the research paper")
if user_input:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retriever_chain = create_retrieval_chain(retriever, document_chain)

    start_time = time.process_time()
    response = retriever_chain.invoke({"input": user_input})
    st.write(f"Response Time: {time.process_time()-start_time}")

    # Display Reponse
    st.write(response['answer'])

    # With Streamlit expander
    with st.expander("Document Similarity Search"):
        for doc in response['context']:
            st.write(doc.page_content)
            st.write('-'*30)