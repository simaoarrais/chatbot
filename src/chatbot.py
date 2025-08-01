import os
import streamlit as st
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import MessagesPlaceholder

from confluence_tool import confluence_tool
from obsidian_tool import obsidian_tool

# Load environment
load_dotenv()
ollama_base_url = os.getenv("OLLAMA_BASE_URL")

# Streamlit UI
st.set_page_config(page_title="LLaMA Chatbot", page_icon="🦙")
st.title("🦙 LLaMA Chatbot")
st.session_state["debug"] = True  # Enables Streamlit debugging in tools

if st.session_state["debug"]:
    session_id = "streamlit-test"

# Define main LLM for reasoning
agent_llm = ChatOllama(
    model="llama3.1:8b", 
    base_url=ollama_base_url,
    temperature=0.7
    )

# Prompt with tool awareness
prompt = ChatPromptTemplate.from_messages([
    ("system", 
        "You are an assistant that answers questions based on tool outputs.\n"
        "You also have access to chat_history — the previous conversation.\n"
        "Use chat_history or tools to track user intent, especially across multiple queries.\n"
        "\n"
        "The tools available are:\n"
        "- **search_confluence**: Search Confluence documents for Counter-Strike lore.\n"
        "- **search_obsidian**: Search Obsidian notes for work-related information.\n"
        "\n"
        "When the user continues a previous topic or adds details (e.g. mentions another source like Obsidian or Confluence),"
        "combine the previous query context from chat_history with the new input to form a more complete tool query.\n"
        "\n"
        "If answering requires checking more than one source or tool, call each tool as needed before answering."
    ),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create agent with tool
tools = [confluence_tool, obsidian_tool]
agent = create_tool_calling_agent(agent_llm, tools, prompt)
executor = AgentExecutor(
    agent=agent, 
    tools=tools,
    verbose=True
)

# Initialize message history
history = StreamlitChatMessageHistory(key="chat_messages")
chain_with_history = RunnableWithMessageHistory(
    executor,
    lambda session_id: history,
    input_messages_key="input",
    history_messages_key="chat_history",
)

if st.button("🗑️ Clear Chat History"):
    history.clear()
    st.rerun()

# Display chat history
for message in history.messages:
    if message.type == "human":
        st.chat_message("human").write(message.content)
    elif message.type == "ai":
        st.chat_message("ai").write(message.content)

# Chat UI
user_input = st.chat_input("Type your message")
if user_input:
    st.chat_message("human").write(user_input)

    with st.spinner("Thinking..."):
        response = chain_with_history.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        st.chat_message("ai").write(response['output'])
