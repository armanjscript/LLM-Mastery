import os
from langchain_community.document_loaders import TextLoader
import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

script_directory = os.path.dirname(os.path.abspath(__file__))

# Initialize the embedding model
embedding = embeddings.OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=8)

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
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 2})

print("Retriever created.")

# Define a user query
user_query = "what do you know of Panagiotis Soutsos"

# Retrieve relevant documents
# docs = retriever.invoke(user_query)

template = """
Answer the question based only on the following context:
{context}

Question: {question}
"""

chat_prompt_template = ChatPromptTemplate.from_template(template=template)

model = OllamaLLM(model="qwen2.5:latest")

#chain = chat_prompt_template | model | StrOutputParser()

rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | chat_prompt_template
    | model
    | StrOutputParser()
)

response = rag_chain.invoke(user_query)

print(response)
