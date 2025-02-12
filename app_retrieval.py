import os
from langchain_community.document_loaders import TextLoader
import pprint
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community import embeddings

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
retriever = vector_store.as_retriever(search_type="similarity",search_kwargs={"k": 2})

print("Retriever created.")

# Define a user query
# user_query = "what do you know of Panagiotis Soutsos"

queries = [
    "what do you know of Panagiotis Soutsos",
    "tell me about Baron Pierre de Coubertin"
]

# Retrieve relevant documents
#Using map() for multiple queries
docs = retriever.map().invoke(queries)

# print(f"Number of documents retrieved: {len(docs)}")
# print(docs)

for query, docs in zip(queries, docs):
    print(f"Results for '{query}':")
    print(f"Number of documents retrieved: {len(docs)}")
    pprint.pprint(docs)
    print("\n" + "-"*40 + "\n")