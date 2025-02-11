from sqlalchemy import create_engine
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import ConfigurableFieldSpec

engine = create_engine("sqlite:///memory.db")

def get_session_history(session_id: str):
    return SQLChatMessageHistory(session_id=session_id, connection=engine)


model = OllamaLLM(model="qwen2.5:latest")
parser = StrOutputParser()

chat_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage(content="You are a helpful assistant. Answer all questions to the best of your ability."),
    MessagesPlaceholder(variable_name="history"),
    HumanMessagePromptTemplate.from_template("{question}")
])

chain = chat_prompt_template | model | parser

runnable_with_history = RunnableWithMessageHistory(
    runnable=chain,
    input_messages_key="question",
    get_session_history=get_session_history,
    history_factory_config=[
        ConfigurableFieldSpec(
            id="session_id",
            annotation=str,
            name="Session ID",
            description="Unique identifier for the conversation session.",
            default="",
            is_shared=True
        ),
    ],
    history_messages_key="history"
)

while True:
    session_id = input("Enter your Session ID (or type 'exit' to quit): ").strip()
    if session_id.lower() == "exit":
        break
    
    user_query = input("Ask anything (or type 'exit' to quit): ").strip()
    if user_query.lower() == "exit":
        break
    
    response = runnable_with_history.invoke(
        {"question": user_query},
        config={"configurable": {"session_id": session_id}}
   )
    
    print(response)



