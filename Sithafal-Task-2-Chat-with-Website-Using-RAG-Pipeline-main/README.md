#Chat-with-Website-Using-RAG-Pipeline

For more details I suggest you to see below video:

Youtube Video: 

Overview

This project implements a Retrieval-Augmented Generation (RAG) Pipeline to allow users to query website content efficiently. 
The system:

1.Crawls and scrapes data from websites.
2.Generates vector embeddings for the website content using a pre-trained model.
3.Stores embeddings in a FAISS vector database.
4.Handles user queries to retrieve relevant data and generate context-aware responses.

Technologies Used:

1.Python
2.BeautifulSoup4 and requests for web scraping.
3.SentenceTransformers for embedding generation.
4.FAISS (Facebook AI Similarity Search) for vector database.
5.Large Language Model (Optional) for response generation.

Directory Structure:

Sithafal_Task_2

├── Task_2/
|   ├── scrape_and_embed.py      # Crawls websites, scrapes data, and generates embeddings
|   ├── query_retrieve.py        # Accepts user queries and retrieves relevant documents
|   ├── generate_response.py     # (Optional) Generates detailed responses using an LLM
|   ├── documents.txt            # Raw scraped website content
|   ├── website_embeddings.index # FAISS index storing embeddings
|   └── retrieved_docs.txt       # Contains top retrieved documents
└──requirements.txt              # Required installations are mentioned in this

How to Run the Project:
Clone the repository:
git clone https://github.com/varalaxmigithubsadula/Sithafal-Task-2-Chat-with-Website-Using-RAG-Pipeline.git

cd Sithafal-Task-2-Chat-with-Website-Using-RAG-Pipeline/Sithafal_Task_2

Install required libraries using pip

pip install -r requirements.txt

Run the following Scripts orderly:

python scrape_and_embed.py
python query_retrieve.py
python generate_response.py

Output Demonstration:

Enter the queries and get the answers

Example Queries

1. What is the university of Chicago known for?

2. Tell me about Stanford University?
   
Below are the Snapshots of the Project's Output:

After Executing python scrape_and_embed.py file

![image](https://github.com/user-attachments/assets/e2ceec42-e2fd-4c49-b9ad-464d08ad651e)

![image](https://github.com/user-attachments/assets/aaff38c5-1dfe-4c07-822d-181edec8f5ff)


