import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Function to convert query to embeddings
def get_embeddings(text, model):
    return model.encode([text], show_progress_bar=False)

# Load FAISS index and SentenceTransformer model
index = faiss.read_index("website_embeddings.index")
model = SentenceTransformer('all-MiniLM-L6-v2')

# Load documents with UTF-8 encoding
try:
    with open("documents.txt", "r", encoding="utf-8") as f:
        documents = f.readlines()
except UnicodeDecodeError as e:
    print("Error reading 'documents.txt'. Ensure it is UTF-8 encoded.")
    raise e

# User query
user_query = input("Enter your query: ")

# Convert query to embeddings
query_embedding = get_embeddings(user_query, model)

# Perform similarity search in FAISS
print("Retrieving relevant documents...")
top_k = 5
_, indices = index.search(query_embedding, top_k)

# Retrieve and save the top documents
relevant_docs = [documents[i].strip() for i in indices[0] if i < len(documents)]
with open("retrieved_docs.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(relevant_docs))

# Display results
print("Top relevant documents retrieved:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"\nDocument {i}:")
    print(doc)
