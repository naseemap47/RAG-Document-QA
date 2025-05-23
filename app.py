import streamlit as st
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
load_dotenv()
import chromadb.api
chromadb.api.client.SharedSystemClient.clear_system_cache()

# HF Embeddings
os.environ['HF_TOKEN'] = os.getenv('HF_TOKEN')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# Streamlit
st.title("Q&A RAG with PDF with Chat History")
st.write("Upload PDF and chat with their content")

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

# Select Session ID
session_id = st.text_input("Session ID", value="default")

# Ollama LLM
llm = OllamaLLM(model=llm_model)

if 'store' not in st.session_state:
    st.session_state.store = {}

# Upload PDF
uploaded_files = st.file_uploader("Upload PDF File", type="pdf", accept_multiple_files=True)
if uploaded_files:
    documents = []
    for uploaded_file in uploaded_files:
        temp_pdf = f"./temp.pdf"
        with open(temp_pdf, 'wb') as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name
        
        loader = PyPDFLoader(temp_pdf)
        docs = loader.load()
        documents.extend(docs)
    
    # Split and create embeddings for the documents
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_spliter.split_documents(docs)
    vector_store = Chroma.from_documents(documents=splits, embedding=embeddings)
    retriever = vector_store.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate if if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # Answer Question Prompt
    system_prompt = (
        "You are an assistant for question-answering task. "
        "use the following pieces of retrieved contex to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentance maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ]
    )

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    def get_session_history(session_id:str) -> BaseChatMessageHistory:
        if session_id not in st.session_state.store:
            st.session_state.store[session_id] = ChatMessageHistory()
        return st.session_state.store[session_id]
    
    conversation_rag_chain = RunnableWithMessageHistory(
        rag_chain, get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    user_input = st.text_input("Ask Question?")
    if user_input:
        session_history = get_session_history(session_id)
        response = conversation_rag_chain.invoke(
            {"input": user_input},
            config={
                "configurable": {"session_id": session_id}
            }
        )

        st.write(st.session_state.store)
        st.success(f"Assistant: {response['answer']}")
        st.write("Chat History: ", session_history.messages)
