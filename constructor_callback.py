from langchain_core.callbacks import BaseCallbackHandler
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage
from pprint import pprint
from langchain_core.outputs import LLMResult
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LoggingHandler(BaseCallbackHandler):
    
    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], **kwargs) -> None:
        print("Chat model started")
        print(f"Messages: {messages}")
        
    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        print("Response:")
        pprint(response)
        
    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        chain_name = serialized.get('name') if serialized and 'name' in serialized else "Unknown Chain"
        print(f"Chain {chain_name} started")
        print("Serialized Variable:")
        pprint(serialized)
        print("Inputs:")
        pprint(inputs)
        
    
    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        print("Outputs:")
        pprint(outputs)
        print("Chain ended")
        
        
callbacks = [LoggingHandler()]
llm = OllamaLLM(model="qwen2.5:latest", callbacks=callbacks)
prompt = ChatPromptTemplate.from_template("What is the square root of {number}?")

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"number": 16})

print("Final response:")
pprint(response)