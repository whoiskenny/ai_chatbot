import os, io
import PyPDF2
import tempfile
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv
from utilz import run_llm, get_faiss_vectordb
from indexing import get_pinecone_vectordb
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import pinecone as Pinecone
from langchain_community.vectorstores.faiss import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from pinecone import Pinecone as pcClient, PodSpec
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.llms.openai import OpenAIChat
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from openai import OpenAI, BadRequestError
from helpers import (
    generate_vectors,
    get_chat_completion,
    extract_text_from_file,
    generate_summary,
    generate_file_chunks,
)



load_dotenv()

TMP_DIR = Path(__file__).resolve().parent.joinpath('data', 'tmp')
LOCAL_VECTOR_STORE_DIR = Path(__file__).resolve().parent.joinpath('data', 'vector_store')

PINECONE_API_KEY = pcClient(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_index = PINECONE_API_KEY.Index(PINECONE_INDEX_NAME)


pinecone = PINECONE_API_KEY
client = OPENAI_API_KEY
index_name = PINECONE_INDEX_NAME

EMBEDDING_MODEL = 'text-embedding-ada-002'

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

if index_name not in [index['name'] for index in pinecone.list_indexes()]:
    pinecone.create_index(
        index_name, 
        dimension=1536,
        metric='cosine',
        environment=os.getenv("PINECONE_ENVIRONMENT")
    )

index = pinecone.Index(index_name)

def process_docs():
    try:
        if 'source_docs' not in st.session_state:
            st.session_state.source_docs = []

        for source_doc in st.session_state.source_docs:
            with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
                tmp_file.write(source_doc.read())

                documents = load_documents()
                texts = split_documents(documents)

                if not st.session_state.pinecone_db:
                    st.session_state.retriever = embeddings_on_local_vectordb(texts)
                else:
                    st.session_state.retriever = embeddings_on_pinecone(texts)
    except Exception as e:
        st.error(f"An error occurred: {e}")

def load_documents(separate_uploader):
    with tempfile.NamedTemporaryFile(delete=False, dir=TMP_DIR.as_posix(), suffix='.pdf') as tmp_file:
        tmp_file.write(separate_uploader.read())
        documents = DirectoryLoader(TMP_DIR.as_posix(), glob='**/*.pdf').load()
    return documents

def split_documents(documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

def embeddings_on_local_vectordb(texts):
    vectordb = Chroma.from_documents(texts, embedding=OpenAIEmbeddings(),
                                    persist_directory=LOCAL_VECTOR_STORE_DIR.as_posix())
    vectordb.persist()
    retriever = vectordb.as_retriever(search_kwargs={'k': 7})
    return retriever

def embeddings_on_pinecone(texts):
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = Pinecone.from_documents(texts, embeddings, index_name=PINECONE_INDEX_NAME)
    retriever = vectordb.as_retriever()
    return retriever


def query_llm(retriever, query):
    if retriever is not None:
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=OpenAIChat(openai_api_key=OPENAI_API_KEY),
            retriever=retriever,
            return_source_documents=True,
        )
        result = qa_chain({'question': query, 'chat_history': st.session_state.messages})
        result = result['answer']
        st.session_state.messages.append((query, result))
        return result
    else:
        st.error("Retriever is not initialized. Please upload a document first.")
        return None

def process_query(question):
    if 'retriever' not in st.session_state:
        st.error("Please upload and submit documents first.")
        return

    retriever = st.session_state.retriever
    query_result = query_llm(retriever, question)
    st.session_state.query_result = query_result


def boot():

    st.set_page_config(page_title="RAG")
    st.title("Retrieval Augmented Generation Engine")

    # Allow the user to upload a file with supported extensions.
    uploaded_file = st.file_uploader("Upload an article", type=("txt", "md", "pdf"))
    st.session_state.source_docs = uploaded_file
    
    st.session_state.pinecone_db = st.checkbox('Use Pinecone Vector DB')    
    st.button("Submit Documents")


    # Provide a text input field for the user to ask questions about the uploaded article.
    question = st.text_input(
        "Ask something about the article",
        placeholder="Can you give me a short summary?",
        disabled=not uploaded_file,
    )


    # If an uploaded file is available, process it.
    if uploaded_file:

        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Create a FAISS vector database from the uploaded file.
        vectordb = get_faiss_vectordb(uploaded_file.name)


        # If the vector database is not created (unsupported file type), display an error message.
        if vectordb is None: #and vectordb2 is None:
            st.error(
                f"The {uploaded_file.type} is not supported. Please load a file in pdf, txt, or md"
            )
        

    # Display a spinner while generating a response.
    with st.spinner("Generating response..."):
        # If both an uploaded file and a question are available, run the model to get an answer.
        if uploaded_file and question:
            answer = run_llm(vectordb=vectordb, query=question)
            # Display the answer in a Markdown header format.
            st.write("### Answer")
            st.write(f"{answer}")

          #
    if "messages" not in st.session_state:
        st.session_state.messages = []


if __name__ == '__main__':
    boot()