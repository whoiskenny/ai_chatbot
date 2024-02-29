#Rag in 4 steps
# Data loading
# Splitting the document
# Generating Vectors
# Getting context and query the LLM 
import os
from openai import OpenAI, BadRequestError
import streamlit as st
import PyPDF2
import pypdf
import dotenv
import tiktoken
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain, RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from pinecone import Pinecone as pcClient, PodSpec
import getpass
from helpers import (
    generate_vectors,
    get_chat_completion,
    extract_text_from_file,
    generate_summary,
    generate_file_chunks,
)

load_dotenv()

PINECONE_API_KEY = pcClient(api_key=os.getenv("PINECONE_API_KEY"))
PINECONE_ENV = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
OPENAI_API_KEY = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pinecone_index = PINECONE_API_KEY.Index(PINECONE_INDEX_NAME)


pinecone = PINECONE_API_KEY
client = OPENAI_API_KEY
index_name = PINECONE_INDEX_NAME

OPENAI_API_KEY= OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def get_faiss_vectordb(file: str):
    # Extract the filename and file extension from the input 'file' parameter.
    filename, file_extension = os.path.splitext(file)

    # Initiate embeddings using OpenAI.
    embedding = OpenAIEmbeddings()

    # Create a unique FAISS index path based on the input file's name.
    faiss_index_path = f"faiss_index_{filename}"

    # Determine the loader based on the file extension.
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=file)
    elif file_extension == ".txt":
        loader = TextLoader(file_path=file)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path=file)
    else:
        # If the document type is not supported, print a message and return None.
        print("This document type is not supported.")
        return None

    # Load the document using the selected loader.
    documents = loader.load()

    # Split the loaded text into smaller chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separators=["\n", "\n\n", "(?<=\. )", "", " "],
    )
    doc_chunked = text_splitter.split_documents(documents=documents)

    # Create a FAISS vector database from the chunked documents and embeddings.
    vectordb = FAISS.from_documents(doc_chunked, embedding)
    
    # Save the FAISS vector database locally using the generated index path.
    vectordb.save_local(faiss_index_path)
    
    # Return the FAISS vector database.
    return vectordb

def run_llm(vectordb, query: str) -> str:
    # Create an instance of the ChatOpenAI with specified settings.
    openai_llm = ChatOpenAI(temperature=0, verbose=True)
    
    # Create a RetrievalQA instance from a chain type with a specified retriever.
    retrieval_qa = RetrievalQA.from_chain_type(
        llm=openai_llm, chain_type="stuff", retriever=vectordb.as_retriever()
    )
    
    # Run a query using the RetrievalQA instance.
    answer = retrieval_qa.run(query)
    
    # Return the answer obtained from the query.
    return answer




def openai_api_key():
    return os.getenv("OPENAI_API_KEY", "default_value_if_not_found")

def setup_pinecone_index(pinecone, index_name):
    pcClient.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        spec=PodSpec(
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
    )


def split_docs(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", "\n\n", "(?<=\. )", "", " "],)
    docs = text_splitter.split_documents(documents)
    return docs

def get_pinecone_vectordb(file: str, index_name: str):
    # Extract the filename and file extension from the input 'file' parameter.
    filename, file_extension = os.path.splitext(file)

    # Initiate embeddings using OpenAI.
    # openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Determine the loader based on the file extension.
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=file)
    elif file_extension == ".txt":
        loader = TextLoader(file_path=file)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path=file)
    else:
        # If the document type is not supported, print a message and return None.
        print("This document type is not supported.")
        return None

    # Load the document using the selected loader.
    documents = loader.load()

    # contents = [doc.content for doc in documents]

    # content = "\n".join(contents)

    # content = preprocess_content(content)

    # Split the loaded text into smaller chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separators=["\n", "\n\n", "(?<=\. )", "", " "],
    )
    doc_chunked = text_splitter.split_documents(documents=documents)

    # Create an empty list to store vectors.
    vectors = []

    try:
        for chunk in doc_chunked:
            try:
                # Generate embeddings for each chunk of text.
                for chunk in doc_chunked:
                    # Generate embeddings for the text chunk using OpenAI.
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo-0125",  # You may adjust the engine according to your needs
                        messages=chunk,
                        max_tokens=50,
                        temperature=0,
                        top_p=1.0,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        stop=["\n"]
                    )
                    # Extract the embeddings from the OpenAI response.
                    embedding = response.choices[0].embedding
                    vectors.append(embedding)
            except BadRequestError as e:
                print(f"Error generating embeddings for chunk: {e}")
                
    except BadRequestError as e:
        print(f"Error processing document chunks: {e}")
        # You may choose to handle the error here, e.g., by returning None or logging the error

    # Create or retrieve the index
    index = pinecone.Index(index_name)

    # Upload vectors to the Pinecone index
    index.upsert(vectors)

    return index

def preprocess_content(content: str) -> str:
    # Implement any preprocessing steps needed to clean and format the content.
    # For example, you may want to remove special characters, extra whitespaces, etc.
    # Here's a simple example to remove non-alphanumeric characters:
    cleaned_content = ''.join(char for char in content if char.isalnum() or char.isspace())
    return cleaned_content


def pinecone_vectorisation(uploaded_file):
    try:
        if not uploaded_file:
            return {"error": "No file provided"}

        # Read the content of the uploaded file
        file_content = uploaded_file.read()

        # Extract text from the file content
        file_text = extract_text_from_file(file_content)

        # Split the text into chunks
        file_docs = generate_file_chunks(file_text)

        # Extract the text content from the chunks
        texts = [doc.page_content for doc in file_docs]

        # Generate vectors for the text
        file_vectors = generate_vectors(texts)

        # Delete all old indexes to prevent overlap
        pinecone_index.delete(delete_all=True)

        # Get the ID of the last vector in the index
        index_stats = pinecone_index.describe_index_stats()
        num_vectors = index_stats['total_vector_count']

        # Generate unique IDs for each vector
        ids = [str(index + 1) for index in range(num_vectors, num_vectors + len(file_vectors))]

        # Upsert vectors into the Pinecone index
        pinecone_index.upsert(vectors=[(id, embedding, {"text": metadata}) for id, embedding, metadata in zip(ids, file_vectors, texts)])

        return {"message": "File uploaded successfully!", "error": False}
    
    except Exception as e:
        return {"message": f"Error while trying to process File: {e}", "error": True}
    
def store_vectors_in_pinecone(file: str, index_name: str = None):
    # Determine the loader based on the file extension.
    filename, file_extension = os.path.splitext(file)
    if file_extension == ".pdf":
        loader = PyPDFLoader(file_path=file)
    elif file_extension == ".txt":
        loader = TextLoader(file_path=file)
    elif file_extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path=file)
    else:
        print("This document type is not supported.")
        return None

    # Load the document using the selected loader.
    documents = loader.load()

    # Split the loaded text into smaller chunks for processing.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=30,
        separators=["\n", "\n\n", "(?<=\. )", "", " "],
    )
    doc_chunked = text_splitter.split_documents(documents=documents)

    # Create an empty list to store vectors.
    vectors = []

    try:
        for chunk in doc_chunked:
            # Generate embeddings for the text chunk using OpenAI.
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                messages=[chunk],
                max_tokens=50,
                temperature=0,
                top_p=1.0,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                stop=["\n"]
            )
            # Extract the embeddings from the OpenAI response and append to the vectors list.
            embedding = response.choices[0].embedding
            vectors.append(embedding)
    except BadRequestError as e:
        print(f"Error generating embeddings: {e}")


    if not vectors:
        print("No vectors generated from the document.")
        return None

    # Create or retrieve the Pinecone index
    pinecone_index = PINECONE_API_KEY.Index(index_name)

    # Upsert vectors to the Pinecone index
    try:
        pinecone_index.upsert(vectors)
        print("Vectors successfully stored in the Pinecone index.")
    except Exception as e:
        print(f"Error upserting vectors to Pinecone: {e}")

    return pinecone_index

def pc_process_vectorize_file(uploaded_file):
    try:
        if not uploaded_file:
            return {"error": "No file provided"}

        # Read the content of the uploaded file
        file_content = uploaded_file.read()

        # Extract text from the file content
        file_text = extract_text_from_file(file_content)

        # Split the text into chunks
        file_docs = generate_file_chunks(file_text)

        # Extract the text content from the chunks
        texts = [doc.page_content for doc in file_docs]

        # Generate vectors for the text
        file_vectors = generate_vectors(texts)

        # Delete all old indexes to prevent overlap
        pinecone_index.delete(delete_all=True)

        # Get the ID of the last vector in the index
        index_stats = pinecone_index.describe_index_stats()
        num_vectors = index_stats['total_vector_count']

        # Generate unique IDs for each vector
        ids = [str(index + 1) for index in range(num_vectors, num_vectors + len(file_vectors))]

        # Upsert vectors into the Pinecone index
        pinecone_index.upsert(vectors=[(id, embedding, {"text": metadata}) for id, embedding, metadata in zip(ids, file_vectors, texts)])

        return {"message": "File uploaded successfully!", "error": False}
    
    except Exception as e:
        return {"message": f"Error while trying to process File: {e}", "error": True}