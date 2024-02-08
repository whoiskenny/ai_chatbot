# indexing.py
import os
import io
import PyPDF2
from pinecone import Pinecone, PodSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

def setup_pinecone_index(pinecone, index_name):
    pinecone.create_index(
        index_name,
        dimension=1536,
        metric='cosine',
        spec=PodSpec(
            environment=os.getenv("PINECONE_ENVIRONMENT")
        )
    )

def create_pinecone_instance():
    return Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


def extract_text_from_file(file):
    """Takes a PDF, extracts the text and converts it into a list of chunks"""
    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file))
    numPages = len(pdf_reader.pages)
    text = ""
    for i in range(numPages):
        page = pdf_reader.pages[i]
        text += page.extract_text()

    text = text.replace("\t", " ")
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[],
        chunk_size=1536,
        chunk_overlap=200,
    )
    
    docs = text_splitter.create_documents([text])
    return docs

#Embeddings and Vector Functions

def get_num_vectors(index):
    index_stats = index.describe_index_stats()
    num_vectors = index_stats['total_vector_count']
    return num_vectors

def generate_vectors(texts):
    """Takes a list of texts, and converts them into embeddings"""
    embeddings_response = client.embeddings.create(input=texts, model=EMBEDDING_MODEL)
    embeddings = [data.embedding for data in embeddings_response.data]
    return embeddings

def query_refiner(conversation, query):
    response = OpenAI.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    return response['choices'][0]['text']

from langchain.document_loaders import DirectoryLoader

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

# Splitting documents
def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        text = ""
        for page_num in range(reader.numPages):
            text += reader.getPage(page_num).extractText()
    return text
