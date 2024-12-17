# Sithafal_tasks
#Chat with PDF using RAG Pipeline
This project allows users to upload PDF files, extract text from the PDFs, and query the document for specific information. It also includes functionality to extract tables from specific pages of the uploaded PDF. The app uses machine learning models to process queries using TF-IDF vectorization and cosine similarity for matching text content.

#Features
PDF Text Extraction: Extracts and processes the text content of uploaded PDF files.
Table Extraction: Extracts tables from specific pages of the uploaded PDF.
Query Processing: Allows users to submit queries, which are matched against the extracted text using TF-IDF and cosine similarity.
File Upload: Provides a simple interface for uploading PDF files.
#Technologies Used
Flask: Web framework for building the application.
PyPDF2: For text extraction from PDF files.
pdfplumber: For extracting tables from PDFs.
Scikit-learn: For TF-IDF vectorization and cosine similarity-based query processing.
HTML/CSS: For frontend design (basic form for uploading PDFs and submitting queries).
#Setup
To run this project locally, follow the instructions below:

#Prerequisites
Ensure that Python 3.x is installed on your machine. You will also need the following Python libraries:

Flask
PyPDF2
pdfplumber
scikit-learn
#Installation Steps
Install the required libraries using pip
pip install -r requirements.txt
Run the app:
Start the Flask development server:

python app.py
Access the app: Open your browser and navigate to http://127.0.0.1:5000/ to view the app.
Below are some snapshots of the output
