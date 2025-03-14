from langchain_community.chat_message_histories import ChatMessageHistory
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

llm = OllamaLLM(model="mistral")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = ChatMessageHistory()

prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="Previous conversation:\n{chat_history}\n\nCurrent question: {question}\n\nResponse: "
)

def run_chain(question):
    chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in st.session_state.chat_history.messages])
    response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
    
    st.session_state.chat_history.add_user_message(question)
    st.session_state.chat_history.add_ai_message(response)
    return response
st.title("🧠 AI Models & Libraries for Different AI Agents")
st.write("Ask me anything")

user_input = st.text_input("Your Question")
if user_input:
    response = run_chain(user_input)
    st.write(f"**Your Question:** {user_input}")
    st.write(f"**AI Response:** {response}")
    
st.subheader("🔥 Chat History")
for msg in st.session_state.chat_history.messages:
    st.write(f"{msg.type.capitalize()}: {msg.content}")


# from langchain_community.chat_message_histories import ChatMessageHistory
# from langchain_core.prompts import PromptTemplate
# from langchain_ollama import OllamaLLM

# llm = OllamaLLM(model="mistral")
# chat_history = ChatMessageHistory()

# prompt = PromptTemplate(
#     input_variables=["chat_history", "question"],
#     template = "Previous conversation:\n{chat_history}\n\nCurrent question: {question}\n\nResponse:",
# )
# def run_chain(question):
#     chat_history_text = "\n".join([f"{msg.type.capitalize()}: {msg.content}" for msg in chat_history.messages])
#     response = llm.invoke(prompt.format(chat_history=chat_history_text, question=question))
#     chat_history.add_user_message(question)
#     chat_history.add_ai_message(response)
#     return response
    
# print("\n Welcome to your Ai Agent! Ask me anything")
# while True:
#     question = input("Your Question (or type 'exit' to stop): ")
#     if question.lower() == 'exit':
#         print("Goodbye!")
#         break
#     response = run_chain(question)
#     print("\n AI Response: ", response)
    
# from langchain_ollama import OllamaLLM
# llm = OllamaLLM(model="mistral")
# print("\n Welcome to your Ai Agent! Ask me anything")
# while True:
#     question = input("Your Question (or type 'exit' to stop): ")
#     if question.lower() == 'exit':
#         print("Goodbye!")
#         break
#     response = llm.invoke(question)
#     print("\n AI Response: ", response)