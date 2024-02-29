import os, re
from openai import OpenAI, BadRequestError
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain, ConversationalRetrievalChain, RetrievalQA
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from pinecone import Pinecone, PodSpec
import getpass

load_dotenv()

pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# EMBEDDING_MODEL = 'text-embedding-ada-002'
# index_name = 'index'

# if index_name not in [index['name'] for index in pinecone.list_indexes()]:
#     pinecone.create_index(
#         index_name, 
#         dimension=1536,
#         metric='cosine',
#         spec=PodSpec(
#             environment=os.getenv("PINECONE_ENVIRONMENT")
#         )
#     )

# index = pinecone.Index(index_name)

def split_docs(documents, chunk_size=500, chunk_overlap=20):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs

def create_openai_instance():
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))



def get_pinecone_vectordb(file: str):
    # Extract the filename and file extension from the input 'file' parameter.
    filename, file_extension = os.path.splitext(file)

    # Initialize embeddings using OpenAI.
    embedding = OpenAIEmbeddings()

    # Initialize Pinecone with your API key

    # Create the index if it doesn't exist
    if index_name not in pinecone.list_indexes():
        create_pinecone_instance()

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

    # Create an empty list to store vectors.
    all_vectors = []

    try:
        for chunk in doc_chunked:
            # Generate embeddings for each chunk of text.
            embeddings = embedding(chunk)
            all_vectors.extend(embeddings)
    except Exception as e:
        print(f"Error generating embeddings: {e}")

    if not all_vectors:
        print("n vectors generated for upsert.")

        # Create or retrieve the index
        index_name = 'index'
        index = pinecone.Index(index_name)

        # Upload vectors to the Pinecone index
        index.upsert(all_vectors)

        return index_name

# def get_pinecone_vectordb(file: str, index_name: str):
#     # Extract the filename and file extension from the input 'file' parameter.
#     filename, file_extension = os.path.splitext(file)

#     # Initiate embeddings using OpenAI.
#     create_pinecone_instance()
#     create_openai_instance()
#     # openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#     embedding = OpenAIEmbeddings()


#     # Determine the loader based on the file extension.
#     if file_extension == ".pdf":
#         loader = PyPDFLoader(file_path=file)
#     elif file_extension == ".txt":
#         loader = TextLoader(file_path=file)
#     elif file_extension == ".md":
#         loader = UnstructuredMarkdownLoader(file_path=file)
#     else:
#         # If the document type is not supported, print a message and return None.
#         print("This document type is not supported.")
#         return None

#     # Load the document using the selected loader.
#     documents = loader.load()

#     # Split the loaded text into smaller chunks for processing.
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=30,
#         separators=["\n", "\n\n", "(?<=\. )", "", " "],
#     )
#     doc_chunked = text_splitter.split_documents(documents=documents)

#     # Create an empty list to store vectors.
#     vectors = []

#     try:
#         for chunk in doc_chunked:
#             vectors = embedding(chunk)
#             Pinecone.insert_items(index_name, vectors)
#             try:
#                 # Generate embeddings for each chunk of text.
#                 for chunk in doc_chunked:
#                     # Generate embeddings for the text chunk using OpenAI.
#                     response = client.chat.completions.create(
#                         model="gpt-3.5-turbo-0125",  # You may adjust the engine according to your needs
#                         messages=chunk,
#                         max_tokens=50,
#                         temperature=0,
#                         top_p=1.0,
#                         frequency_penalty=0.0,
#                         presence_penalty=0.0,
#                         stop=["\n"]
#                     )
#                     # Extract the embeddings from the OpenAI response.
#                     embedding = response.choices[0].embedding
#                     vectors.append(embedding)
#             except BadRequestError as e:
#                 print(f"Error generating embeddings for chunk: {e}")
                
#     except BadRequestError as e:
#         print(f"Error processing document chunks: {e}")
#         # You may choose to handle the error here, e.g., by returning None or logging the error

#     # Create or retrieve the index
#     index = pinecone.Index(index_name)

#     # Upload vectors to the Pinecone index
#     index.upsert(vectors)

#     return index_name

def create_pinecone_instance():
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# def get_pinecone_vectordb(file: str, index_name: str):
#     # Initialize Pinecone with your API key
#     create_pinecone_instance()
    
#     # Extract the filename and file extension from the input 'file' parameter.
#     filename, file_extension = os.path.splitext(file)

#     # Initiate embeddings using OpenAI.
#     embedding = OpenAIEmbeddings()

#     # Create a unique Pinecone index name based on the input file's name.
#     index_name = f"{filename}"

#     # Determine the loader based on the file extension.
#     if file_extension == ".pdf":
#         loader = PyPDFLoader(file_path=file)
#     elif file_extension == ".txt":
#         loader = TextLoader(file_path=file)
#     elif file_extension == ".md":
#         loader = UnstructuredMarkdownLoader(file_path=file)
#     else:
#         # If the document type is not supported, print a message and return None.
#         print("This document type is not supported.")
#         return None

#     # Load the document using the selected loader.
#     documents = loader.load()

#     # Split the loaded text into smaller chunks for processing.
#     text_splitter = RecursiveCharacterTextSplitter(
#         chunk_size=1000,
#         chunk_overlap=30,
#         separators=["\n", "\n\n", "(?<=\. )", "", " "],
#     )
#     doc_chunked = text_splitter.split_documents(documents=documents)

#     # Insert vectors into the Pinecone index
#     for chunk in doc_chunked:
#         vectors = embedding(chunk)
#         Pinecone.insert_items(index_name, vectors)
    
#     # Return the Pinecone index name
#     return index_name

# Define a function to preprocess text
def preprocess_text(text):
    # Replace consecutive spaces, newlines and tabs
    text = re.sub(r'\s+', ' ', text)
    return text


def process_pdf(file_path):
    # create a loader
    loader = PyPDFLoader(file_path)
    # load your data
    data = loader.load()
    # Split your data up into smaller documents with Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(data)
    # Convert Document objects into strings
    texts = [str(doc) for doc in documents]
    return texts

# Define a function to create embeddings
def create_embeddings(texts):
    embeddings_list = []
    for text in texts:
        res = client.embeddings.create(input=[text], engine=MODEL)
        embeddings_list.append(res['data'][0]['embedding'])
    return 


# Define a function to upsert embeddings to Pinecone
def upsert_embeddings_to_pinecone(index, embeddings, ids):
    index.upsert(vectors=[(id, embedding) for id, embedding in zip(ids, embeddings)])
