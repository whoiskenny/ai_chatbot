import io
import os
import PyPDF2
from pinecone import Pinecone, PodSpec
from openai import OpenAI
from dotenv import load_dotenv
import streamlit as st
from streamlit_chat import message
import google.generativeai as genai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains import ConversationalRetrievalChain   
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.vectorstores import FAISS   
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from indexing import (
    setup_pinecone_index, 
    get_num_vectors, 
    get_num_vectors, 
    generate_vectors,
    query_refiner,
    extract_text_from_file,
    create_pinecone_instance,
    load_docs,
    split_docs,
    extract_text_from_pdf
)

load_dotenv()

#API keys
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = OpenAI(api_key="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")

# Declare embedding model
EMBEDDING_MODEL = 'text-embedding-ada-002'

llm = ChatOpenAI(model_name="gpt-3.5-turbo", 
openai_api_key=os.getenv("OPENAI_API_KEY"))

# global variable for openai to store memory of chat
gpt_msg = [
    {
        "role": "system", 
        "content": 
        """
        #You have been provided a statement of work, I want you to answer a range of questions regarding
        to the document and to provide information that user requests for. The user will provide you new information and I want you to keep meory of it.
        Some examples could be: "I would like to add Lewis, Kobe and Tony to the project", you should be able to keep memory of the users input.

        """
    }
]

def get_openai_response(user_message):
    try:
        new_messages = gpt_msg.copy()
        new_messages.append({"role": "user", "content": user_message})

        response = client.chat.completions.create(
            model='gpt-3.5-turbo',
            messages=new_messages,
            temperature=0, 
        )

        # Store assistant message
        assistant_message = response.choices[0].message
        gpt_msg.append(user_message)
        gpt_msg.append(assistant_message)

        # # Convert ChatCompletionMessage to dictionary
        # assistant_message_dict = {
        #     "content": assistant_message.content,
        #     "role": assistant_message.role,
        #     # Add other attributes you want to include
        # }

        # return assistant_message_dict["content"]

         # Store assistant message
        assistant_message = response.choices[0].message
        gpt_msg.append(user_message)
        gpt_msg.append(assistant_message)

        return assistant_message.content

    except Exception as e:
        print(f"Error while attempting to generate response using GPT: {e}")


# Configure all API keys
pinecone = create_pinecone_instance()

index_name = "app-index"

if index_name not in [index['name'] for index in pinecone.list_indexes()]:
    setup_pinecone_index(pinecone, index_name)

#open file
# extract_text_from_file()

# directory = 'data'

# documents = load_docs(directory)
# docs = split_docs(documents)


# # Creating embeddings
# from langchain.embeddings import SentenceTransformerEmbeddings
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# query_result = embeddings.embed_query("Hello world")

# #Storing embeddings in Pinecone 
# from langchain.vectorstores import Pinecone
# index = Pinecone.from_documents(docs, embeddings, index_name=index_name)

#FRONTEND
st.title("JARVIS")

# Streamlit container for chat history & textbox
response_container = st.container()
text_container = st.container()

# Initialize session state
if 'responses' not in st.session_state:
    st.session_state['responses'] = []

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

if 'buffer_memory' not in st.session_state:
            st.session_state.buffer_memory=ConversationBufferWindowMemory(k=3,return_messages=True)


# system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
# and if the answer is not contained within the text below, say 'I don't know'""")
# human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
# prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])
# conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)
            
#define chat prompt template
system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know'""")
human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

conversation = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm, verbose=True)

# Define functions for interacting with the chatbot
def process_user_query(user_message):
    response = get_openai_response(user_message)
    st.session_state.requests.append(user_message)
    st.session_state.responses.append(response)
    return response

def conversational_chat(query):  
    result = chain({"question": query,   
    "chat_history": st.session_state['history']})  
    st.session_state['history'].append

with text_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("typing..."):
            response = process_user_query(query)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            bot_response = st.session_state['responses'][i]
            human_request = st.session_state["requests"][i] if i < len(st.session_state['requests']) else ""
            st.write(f"User: {human_request}")
            st.write(f"Jarvis: {bot_response}")

st.title("PDF Text Extractor")

uploaded_file = st.file_uploader("Upload file")

if uploaded_file is not None:
    # Save the uploaded file to a temporary directory
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.write("File uploaded successfully.")

    # Extract text from the uploaded PDF file
    extracted_text = extract_text_from_pdf(file_path)
