import os
import tempfile
from pathlib import Path
import streamlit as st
from pinecone import Pinecone as pcClient, PodSpec
from dotenv import load_dotenv
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma, Pinecone
from langchain.llms.openai import OpenAIChat
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set up file paths
TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

# Initialize Pinecone API key, environment, and index name from environment variables
PINECONE_API_KEY = pcClient(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize OpenAI API key from environment variable
OPENAI_API_KEY = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_index = PINECONE_API_KEY.Index(PINECONE_INDEX_NAME)

EMBEDDING_MODEL = 'text-embedding-ada-002'
embedding_ = OpenAIEmbeddings(openai_api_key=str(os.getenv("OPENAI_API_KEY")), model=EMBEDDING_MODEL)


pinecone = PINECONE_API_KEY
client = OPENAI_API_KEY
index_name = PINECONE_INDEX_NAME

# Set Streamlit page config
st.set_page_config(page_title="Retrieval Augmented Generation Engine")
st.title("Retrieval Augmented Generation Engine")

# Function to load documents from directory
def load_documents():
    loader = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf')
    documents = loader.load()
    return documents

# Function to split documents into text chunks
def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# Function to initialize embeddings using local vector DB
def embeddings_on_local_vectordb(texts):
    embeddings = embedding_
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

# Function to initialize embeddings using Pinecone
def embeddings_on_pinecone(texts):
    embeddings = embedding_
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=PINECONE_INDEX_NAME)
    retriever = vectordb.as_retriever()
    return retriever

# Function to perform LL model query
def query_llm(retriever, query):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=OpenAIChat(openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
    result = result['answer']
    st.session_state.messages.append((query, result))
    return result

# Function to handle input fields and document submission
def boot():
    st.session_state.pinecone_db = st.toggle('Use Pinecone Vector DB')
    st.session_state.source_docs = st.file_uploader(label="Upload Documents", type="pdf", accept_multiple_files=True)
    
    # Handle document submission
    if st.button("Submit Documents"):
        if not pinecone or not client or not index_name or not st.session_state.source_docs:         
            st.warning(f"Please upload the documents and provide the missing fields.")
        else:
            try:
                for source_doc in st.session_state.source_docs:
                    with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                        tmp_file.write(source_doc.read())
                    
                documents = load_documents()
                for _file in TMP_DIR.iterdir():
                    temp_file = TMP_DIR.joinpath(_file)
                    temp_file.unlink()
                
                texts = split_documents(documents)
                
                if not st.session_state.pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.session_state.retriever = embeddings_on_pinecone(texts)
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Display chat history and handle user queries
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message('human').write(message[0])
        st.chat_message('ai').write(message[1])
    
    if "retriever" in st.session_state:
        if query := st.chat_input():
            st.chat_message("human").write(query)
            response = query_llm(st.session_state.retriever, query)
            st.chat_message("ai").write(response)
    else:
        st.warning("please upload documents to init the retriever.")

if __name__ == '__main__':
    boot()
