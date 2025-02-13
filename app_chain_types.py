import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_community import embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM
from langchain.chains.question_answering import load_qa_chain

embeddings = embeddings.OllamaEmbeddings(model="nomic-embed-text:latest", num_gpu=8)

def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return ' '.join([doc.page_content for doc in documents])

def split_text_into_chunks(text: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.create_documents([text])

def create_vector_store(chunks):
    return Chroma.from_documents(chunks, embeddings)

def generate_response(vector_store, query):
    llm = OllamaLLM(model="qwen2.5:latest")
    chain = load_qa_chain(llm, chain_type="stuff", verbose=True)
    matching_docs = vector_store.similarity_search(query)
    response = chain.invoke(
        {
        "input_documents": matching_docs, 
        "question": query
        }
    )
    
    return response
    

def main():
    st.title("PDF Question-Answering App Using Load QA Chain")
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    data_folder = ""
    default_file_name = "mycv.pdf"
    default_file_path = os.path.join(script_directory, data_folder, default_file_name)
    
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        uploaded_file_path = os.path.join(script_directory, uploaded_file.name)
        with open(uploaded_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    else:
        uploaded_file_path = default_file_path
        st.write(f"No file uploaded. Using default file: {os.path.basename(uploaded_file_path)}")
        
    if os.path.exists(uploaded_file_path):
        raw_text = extract_text_from_pdf(uploaded_file_path)
        
        if raw_text:
            st.write("Splitting text into chunks...")
            chunks = split_text_into_chunks(raw_text)
            
            # Create vector store
            st.write("Creating vector store...")
            vector_store = create_vector_store(chunks)
            
            # User input for query
            query = st.text_input("Enter your query:")
            
            if query:
                st.write("Processing your query...")
                answer = generate_response(vector_store, query)
                st.write("Answer:", answer)
    else:
        st.error(f"File not found: {uploaded_file_path}")

if __name__ == "__main__":
    main()