import os
import requests
from bs4 import BeautifulSoup
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

def scrape_website(url):
    """
    Scrape and clean meaningful content from a website.
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = ""

    # Extract <p>, <h1>, <h2>, <h3> and clean text
    for paragraph in soup.find_all(['p', 'h1', 'h2', 'h3']):
        text = paragraph.get_text(strip=True)  # Strip whitespace
        if text and not text.lower().startswith(("copyright", "©", "helpful links")):
            content += text + "\n"
    return content

def save_documents_to_file(documents, file_name="documents.txt"):
    """
    Save all documents to a file.
    """
    with open(file_name, 'w', encoding='utf-8') as f:
        for doc in documents:
            f.write(doc + "\n")

def load_documents_from_file(file_name="documents.txt", min_length=10):
    """
    Load documents from a file and filter out irrelevant/short lines.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines() if len(line.strip()) > min_length]

def create_faiss_index(embeddings, dimension):
    """
    Create a FAISS index with the provided embeddings.
    """
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def main():
    # Replace with your desired list of URLs
    urls = [
        "https://en.wikipedia.org/wiki/University_of_Chicago",
        "https://en.wikipedia.org/wiki/Stanford_University",
        "https://www.washington.edu"
    ]
    
    # Scrape and save documents
    print("Scraping websites and cleaning content...")
    all_documents = []
    for url in urls:
        content = scrape_website(url)
        print(f"Scraped {len(content)} characters from {url}")
        all_documents.append(content)

    print("Saving documents to file...")
    save_documents_to_file(all_documents)

    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Load and clean documents
    documents = load_documents_from_file()
    print(f"Loaded {len(documents)} cleaned documents.")
    
    # Generate embeddings
    print("Generating embeddings...")
    embeddings = np.array([model.encode(doc) for doc in documents], dtype='float32')
    
    # Build and save FAISS index
    print("Creating FAISS index...")
    index = create_faiss_index(embeddings, embeddings.shape[1])
    faiss.write_index(index, "website_embeddings.index")
    
    print("FAISS index successfully created and saved as 'website_embeddings.index'.")
    print("Embedding generation complete!")

if __name__ == "__main__":
    main()
