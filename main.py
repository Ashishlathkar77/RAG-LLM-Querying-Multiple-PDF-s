# Import necessary libraries
from pypdf import PdfReader
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
import streamlit as st 
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.core.response.pprint_utils import pprint_response
import openai

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_directory):
    pdf_texts = {}
    for filename in os.listdir(pdf_directory):
        if filename.endswith('.pdf'):
            filepath = os.path.join(pdf_directory, filename)
            reader = PdfReader(filepath)
            text = ''
            for page in reader.pages:
                text += page.extract_text()
            pdf_texts[filename] = text
    return pdf_texts

# Directory containing the PDFs
pdf_directory = '/workspaces/RAG-LLM-Querying-Multiple-PDF-s/pdfs'

# Extract text from PDFs and print the first 500 characters of each
pdf_texts = extract_text_from_pdfs(pdf_directory)
for filename, text in pdf_texts.items():
    print(f'Extracted text from {filename}:')
    print(text[:500])  # Print the first 500 characters

# Load OpenAI API key from config
from config import openai_api_key
os.environ['OPENAI_API_KEY'] = openai_api_key

# Directories for persistence and PDFs
PERSIST_DIR = "./storage"
PDF_DIR = "/workspaces/RAG-LLM-Querying-Multiple-PDF-s/pdfs"

# Create the storage directory if it doesn't exist
os.makedirs(PERSIST_DIR, exist_ok=True)

# Check if the index exists, if not, create a new one
if not os.path.exists(os.path.join(PERSIST_DIR, "docstore.json")):
    documents = SimpleDirectoryReader(PDF_DIR).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
    print("Index created and stored!")
else:
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)
    print("Existing index loaded!")

print("Index is ready!")

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create retriever and query engine
retriever = VectorIndexRetriever(index=index, similarity_top_k=4)
postprocessor = SimilarityPostprocessor(similarity_cutoff=0.80)
query_engine = RetrieverQueryEngine(retriever=retriever, node_postprocessors=[postprocessor])

# Example query
query = "What is attention is all you need?"
response = query_engine.query(query)

# Print the response with source
pprint_response(response, show_source=True)
print(response)

# Streamlit app
st.title("PDF Query Interface")
st.write("Enter your query below:")

query_text = st.text_input("Query:")

if st.button("Submit"):
    if query_text:
        query_engine = index.as_query_engine()
        response = query_engine.query(query_text)
        st.write("Response:")
        st.write(response)
    else:
        st.write("Please enter a query.")
