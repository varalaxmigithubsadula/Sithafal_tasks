import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Function to scrape website content
def scrape_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    content = ""
    for paragraph in soup.find_all(['p', 'h1', 'h2', 'h3']):
        content += paragraph.get_text() + "\n"
    return content

# Function to convert text to embeddings
def get_embeddings(text, model):
    return model.encode([text], show_progress_bar=True)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# List of websites to scrape
urls = [
    "https://www.uchicago.edu/",
    "https://www.washington.edu/",
    "https://www.stanford.edu/"
]

# Scrape content from each website and store in documents.txt
documents = []
for url in urls:
    print(f"Scraping content from: {url}")
    content = scrape_website(url)
    documents.append(content)

# Save scraped documents to a file
with open("documents.txt", "w", encoding="utf-8") as f:
    for doc in documents:
        f.write(doc + "\n")


# Generate embeddings and store in FAISS
print("Generating embeddings...")
embeddings = np.vstack([get_embeddings(doc, model) for doc in documents])
dimension = embeddings.shape[1]

print("Storing embeddings in FAISS...")
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Save the FAISS index
faiss.write_index(index, "website_embeddings.index")
print("FAISS index created and saved.")
