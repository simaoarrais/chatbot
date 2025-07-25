import os
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Environment Setup
load_dotenv()
confluence_api_key = os.getenv("CONFLUENCE_API_KEY")
confluence_space_key = os.getenv("CONFLUENCE_SPACE_KEY")
confluence_url = os.getenv("CONFLUENCE_URL")
confluence_username = os.getenv("CONFLUENCE_USERNAME")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

# Load Documents from Confluence
loader = ConfluenceLoader(
    url=confluence_url,
    username=confluence_username,
    api_key=confluence_api_key,
    space_key=confluence_space_key,
    include_attachments=True,
    limit=50
)
documents = loader.load()
st.write(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs_split = text_splitter.split_documents(documents)

embedding_model = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)

vectorstore = FAISS.from_documents(docs_split, embedding_model)
retriever = vectorstore.as_retriever()

# Initialize LLaMA Model
llm = ChatOllama(
    model="llama3.1:8b",
    base_url = ollama_base_url
)

# Setup Chat History & Prompt
history = StreamlitChatMessageHistory(key="chat_messages")

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert in Counter Strike.\n"
        "Use the following context from internal Confluence documents to answer the user's question accurately.\n"
        "You are a chatbot responsible for summarizing confluence articles.\n"
        "Context:\n{context}\n"
    ),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{question}")
])

def get_context(query):
    docs = retriever.get_relevant_documents(query)
    return "\n\n".join(doc.page_content for doc in docs)

chain = prompt | llm

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: history,
    input_messages_key="question",
    history_messages_key="history"
)

# Streamlit Setup
st.set_page_config(page_title="LLaMA", page_icon="🦙")
st.title("🦙 LLaMA Chatbot")
session_id = "streamlit-test"
# session_id = st.session_state.get("user_id", "default-session")

# Chat Interaction
user_input = st.chat_input("Type your message")

if user_input:
    context = get_context(user_input)

    inputs = {
        "context": context,
        "question": user_input,
    }

    response = chain_with_history.invoke(
        inputs,
        config={"configurable": {"session_id": session_id}}
    )

    with st.expander("Retrieved Context (Confluence)"):
        st.markdown(context)

    # st.write("DEBUG Prompt:", prompt)
    # st.write("DEBUG Model:", response)
    # st.write("Chat history:", st.session_state.messages)

if st.button("🗑️ Clear Chat History"):
    history.clear()
    st.rerun()

# Display Chat History
def get_role_label(msg):
    if isinstance(msg, HumanMessage):
        return "🧑‍💻 You"
    elif isinstance(msg, AIMessage):
        return "🦙 LLaMA"
    return ""

for msg in history.messages:
    role = get_role_label(msg)
    if role:
        st.markdown(f"**{role}:** {msg.content}")
