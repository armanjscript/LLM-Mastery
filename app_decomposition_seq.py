import os
from langchain_community.document_loaders import TextLoader
import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

script_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize the embedding model
embedding = embeddings.OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=4)

# Define the path to the data file
data_folder = 'data'
file_name = 'Olympic_History_Part_1.txt'
file_path = os.path.join(script_directory, data_folder, file_name)

# Load the document
loader = TextLoader(file_path=file_path, encoding='utf-8')
data_docs = loader.load()

print("Document loaded successfully.")
print(f"Number of documents: {len(data_docs)}")

# Split the document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False
)

chunks = text_splitter.split_documents(data_docs)

print("Document split into chunks.")
print(f"Number of chunks: {len(chunks)}")

# Initialize the Chroma vector store with the chunks and embeddings
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embedding
)

print("Chroma vector store initialized.")

# Create a retriever from the vector store
retriever = vector_store.as_retriever()

template = """You are a helpful assistant that generates multiple sub-questions related to an input question. \n 
The goal is to break down the input into a set of sub-problems / sub-questions that can be answered in isolation. \n 
Generate multiple search queries related to: {question} \n
Output (3 queries):"""

prompt_decomposition = ChatPromptTemplate.from_template(template)

generate_queries_decomposition = (
    prompt_decomposition
    | OllamaLLM(model="qwen2.5:latest", temperature=0)
    | StrOutputParser()
    | (lambda x: x.split("\n"))
)

user_query = "what do you know of Panagiotis Soutsos"
questions = generate_queries_decomposition.invoke({"question": user_query})
# pprint.pprint(questions)

template = """Here is the question you need to answer:

\n --- \n {question} \n --- \n

Here is any available background question + answer pairs:

\n --- \n {q_a_pairs} \n --- \n

Here is additional context relevant to the question:

\n --- \n {context} \n --- \n

Use the above context and any background question + answer pairs to answer the question: \n {question}
"""

decomposition_prompt = ChatPromptTemplate.from_template(template)

def format_qa_pair(question, answer):
    formatted_string = f"Question: {question}\nAnswer: {answer}\n\n"
    return formatted_string.strip()

q_a_pairs = ""
for q in questions:
    retrieved_docs = retriever.invoke(q)
    
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    inputs = {
        "context": context,
        "question": q,
        "q_a_pairs": q_a_pairs
    }
    
    rag_chain = (
        decomposition_prompt
        | OllamaLLM(model="qwen2.5:latest", temperature=0)
        | StrOutputParser()
    )
    
    answer = rag_chain.invoke(inputs)
    
    q_a_pair = format_qa_pair(q, answer)
    q_a_pairs += "\n---\n" + q_a_pair
    
print("------final answer")
print(q_a_pairs)