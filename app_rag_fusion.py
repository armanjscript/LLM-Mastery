from langchain_community.document_loaders import TextLoader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from langchain.load import dumps, loads
from operator import itemgetter

embedding = embeddings.OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=8)
model = OllamaLLM(model="qwen2.5:latest")


script_directory = os.path.dirname(os.path.abspath(__file__))

data_folder = "data"
file_name = "Olympic_History_Part_1.txt"

file_path = os.path.join(script_directory, data_folder, file_name)

loader = TextLoader(file_path, encoding="utf-8")
data_docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(data_docs)

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding
)

retriever =vector_store.as_retriever()

template = """You are an AI language assistant. Your task is to generate five
different versions of the given user question to retrieve relevant documents from a vector
database. By generating multiple perspectives on the user question, your goal is to help 
the user overcome some of the limitations of the distance-based similarity search.
Provide these alternative questions seperated by newlines. 
Original question: {question}
Additional context: {additional_context}
"""

prompt_options = ChatPromptTemplate.from_template(template)

generate_queries = prompt_options | model | StrOutputParser() | (lambda x: x.split("\n"))

def reciprocal_rank_fusion(results, k=60):
    fused_scores = {}
    
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1/(rank + k)
    
    reranked_results = [
        (loads(doc), score)
        for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    
    return reranked_results


retrieval_chain = generate_queries | retriever.map() | reciprocal_rank_fusion

template = """Answer the following question based on this context:

{context}
Additional Context: {additional_context}

User_Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

llm = OllamaLLM(model="qwen2.5:latest", temperature=0)

final_rag_chain = (
    {
        "context": retrieval_chain,
        "question": itemgetter("question"),
        "additional_context": itemgetter("additional_context")
    } 
    | prompt
    | llm
    | StrOutputParser()
)

user_query = "what do you know of Panagiotis Soutsos"
response = final_rag_chain.invoke({"question": user_query, "additional_context": "Focus on his contributions to modern Greek literature."})
print(response)


