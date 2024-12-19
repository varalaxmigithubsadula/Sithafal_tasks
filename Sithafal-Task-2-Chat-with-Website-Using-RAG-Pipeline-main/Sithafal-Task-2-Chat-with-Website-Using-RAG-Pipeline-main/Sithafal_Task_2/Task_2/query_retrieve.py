import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def load_documents_from_file(file_name="documents.txt", min_length=10):
    """
    Load documents from a file and filter out irrelevant/short lines.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if len(line.strip()) > min_length]

def load_faiss_index(index_file="website_embeddings.index"):
    """
    Load the FAISS index from a file.
    """
    return faiss.read_index(index_file)

def query_embedding(model, query):
    """
    Generate an embedding for the input query using a SentenceTransformer model.
    """
    return np.array([model.encode(query)])

def search_index(index, query_emb, k=5):
    """
    Search the FAISS index with the query embedding and return top results.
    """
    distances, indices = index.search(query_emb, k)
    return distances[0], indices[0]

def main():
    # Load the SentenceTransformer model and FAISS index
    model = SentenceTransformer('all-MiniLM-L6-v2')
    index = load_faiss_index()
    
    # Load documents with filtering
    documents = load_documents_from_file()

    print("Enter your query (or type 'exit' to quit): ", end="")
    while True:
        query = input()
        if query.lower() == "exit":
            break
        
        print("Generating query embedding...")
        query_emb = query_embedding(model, query)
        
        print("Retrieving relevant documents...\n")
        distances, indices = search_index(index, query_emb)
        
        print("Top relevant documents retrieved:\n")
        for i, (dist, idx) in enumerate(zip(distances, indices)):
            if idx != -1 and idx < len(documents):  # Valid document index
                print(f"Document {i+1} (Score: {dist:.4f}):")
                print(documents[idx][:300])  # Show first 300 characters
                print()
            else:
                print(f"Document {i+1}: No valid result.")
        print("Enter your query (or type 'exit' to quit): ", end="")

if __name__ == "__main__":
    main()
