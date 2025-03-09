import json
from pathlib import Path
import streamlit as st
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from openai import OpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
import os
import hmac
import io
from audio_recorder_streamlit import audio_recorder


def check_password():
    """Returns `True` if the user had a correct password."""

    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets[
            "passwords"
        ] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the username or password.
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Return True if the username + password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show inputs for username + password.
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False

if not check_password():
    st.stop()




conversation_placeholder = st.empty()

API_KEY = st.secrets.["auth_key"]

def clear_chat():
    chat_history.clear()
    st.session_state.messages = []
    st.session_state.context_history = []
    # Delete the audio file if it exists
    audio_file_path = "audio_file.wav"
    if os.path.exists(audio_file_path):
        try:
            os.remove(audio_file_path)
        except PermissionError:
            st.warning("Failed to delete audio file. It's being used by another process.")

    conversation_placeholder.empty()
    st.success("Chat history cleared. You can start a new conversation.")
    st.experimental_rerun()


def transcribe_audio_to_text(audio_bytes):
    client = OpenAI(api_key=API_KEY)
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_bytes)
    return transcript.text

def chat_completion_call(text):
    client = OpenAI(api_key=API_KEY)
    messages = [{"role": "user", "content": text}]
    response = client.chat.completions.create(model="gpt-3.5-turbo-1106", messages=messages)
    return response.choices[0].message.content

# def text_to_speech_ai(speech_file_path, api_response):
#     client = OpenAI(api_key=API_KEY)
#     response = client.audio.speech.create(model="tts-1-hd", voice="nova", input=api_response)
#     response.stream_to_file(speech_file_path)

class Document:
    def __init__(self, page_content, metadata):
        if page_content is None or page_content == "":
            raise ValueError("page_content must be a non-empty string")
        self.page_content = page_content
        self.metadata = metadata

def load_documents_from_directory(directory: str) -> List[Document]:
    documents = []
    for file_path in Path(directory).glob("*.json"):
        with open(file_path, "r") as file:
            data = json.load(file)
            for product_data in data:
                product_name = product_data.get("PRODUCT NAME", "")
                product_brand = product_data.get("PRODUCT BRAND", "")
                product_price = product_data.get("PRODUCT PRICE", "")
                product_category = product_data.get("PRODUCT CATEGORY", "")
                product_description = product_data.get("PRODUCT DESCRIPTION", "")
                product_size = product_data.get("PRODUCT SIZE", "")
                product_color = product_data.get("PRODUCT COLOR", "")
                product_link = product_data.get("PRODUCT LINK", "")

                text = f"PRODUCT NAME: {product_name}\n" \
                       f"PRODUCT BRAND: {product_brand}\n" \
                       f"PRODUCT PRICE: {product_price}\n" \
                       f"PRODUCT CATEGORY: {product_category}\n" \
                       f"PRODUCT DESCRIPTION: {product_description}\n" \
                       f"PRODUCT SIZE: {product_size}\n" \
                       f"PRODUCT COLOR: {product_color}\n" \
                       f"PRODUCT LINK: {product_link}\n"

                document = Document(page_content=text, metadata={"source": product_link, "title": product_name})
                documents.append(document)
    return documents

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

st.title('🤖 RAG OpenAI Chatbot')

#openai_api_key = st.secrets.get('OPENAI_API_KEY')
if not API_KEY:
    API_KEY = st.text_input('Enter OpenAI API token:', type='password')
    if not (API_KEY.startswith('sk-') and len(API_KEY) == 51):
        st.warning('Please enter your credentials!')
        st.stop()
    st.secrets['OPENAI_API_KEY'] = API_KEY
else:
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

data_directory = "./data/"

if 'loaded_documents' not in st.session_state:
    loaded_documents = load_documents_from_directory(data_directory)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    texts = text_splitter.split_documents(loaded_documents)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-large')

    persist_directory = 'chroma'
    if not os.path.exists(persist_directory):
        st.success("Database persisted!")
        vectordb = Chroma.from_documents(documents=texts,
                                         embedding=embeddings,
                                         persist_directory=persist_directory)
        vectordb.persist()
    else:
        st.success("Database loaded!")
        vectordb = Chroma(persist_directory=persist_directory,
                          embedding_function=embeddings)

    st.session_state.loaded_documents = loaded_documents
    st.session_state.vectordb = vectordb
    st.session_state.context_history = []
else:
    loaded_documents = st.session_state.loaded_documents
    vectordb = st.session_state.vectordb

qa_system_prompt = """As a clothing store Assistant, your role is to provide helpful responses based on the provided context. Please ensure that your responses are relevant to the user's questions and use the context data effectively.

When answering questions, consider the following guidelines:
- Only provide answers to the user's questions.
- Use the context data provided to generate responses.
- Aim to be informative and helpful.
- For gender-specific questions, consider both the 'PRODUCT TITLE' and 'PRODUCT CATEGORY' for a more accurate response.

For example, if the user asks about available products, you can list the available products along with their details such as brand, category, price, etc.
If the user asks about the most expensive product, identify and provide information about the product with the highest price.
For queries about the cheapest product, identify and provide information about the product with the lowest price.

Below is the context data available for generating responses:
<context>
{context}
</context>

Please keep these guidelines in mind when responding to user queries. If you need clarification or assistance, feel free to ask.
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

contextualize_q_system_prompt = """Given the chat history and the latest user question,
rephrase the user's question into a standalone question that does not rely on previous context.
The reformulated question should be understandable without referencing the chat history.
Do NOT answer the question, only provide a rephrased version if necessary.
Otherwise, return the question as is."""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo")

contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5})

chat_history = st.session_state.context_history if 'context_history' in st.session_state else []

rag_chain = (
        RunnablePassthrough.assign(
            context=contextualized_question | retriever | format_docs
        )
        | qa_prompt
        | llm
        | StrOutputParser()
)


audio_bytes = audio_recorder()
prompt = st.chat_input("What is up?")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        ai_msg = rag_chain.invoke({"question": prompt, "chat_history": chat_history})
        chat_history.append(HumanMessage(content=prompt))
        chat_history.append(ai_msg)
        st.markdown(ai_msg)
    st.session_state.messages.append({"role": "assistant", "content": ai_msg})

elif audio_bytes:
    try:
        # Process audio input
        audio_location = "audio_file.wav"
        with open(audio_location, "wb") as f:
            f.write(audio_bytes)

        # Transcribe audio to text
        text = transcribe_audio_to_text(open(audio_location, "rb"))

        st.session_state.messages.append({"role": "user", "content": text})
        with st.chat_message("user"):
            st.markdown(text)  # Display transcribed text from audio

        with st.chat_message("assistant"):
            ai_msg = rag_chain.invoke({"question": text, "chat_history": chat_history})
            chat_history.append(HumanMessage(content=text))  # Append transcribed text to chat history
            chat_history.append(ai_msg)
            st.markdown(ai_msg)
        st.session_state.messages.append({"role": "assistant", "content": ai_msg})
    except:
        st.error('Audio Too Short!Try again', icon="🚨")

if st.button("New Chat"):
    clear_chat()
