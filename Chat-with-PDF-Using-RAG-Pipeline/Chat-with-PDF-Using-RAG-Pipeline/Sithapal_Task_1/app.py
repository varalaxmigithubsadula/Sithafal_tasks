import re
from flask import Flask, request, jsonify, render_template
import os
import pdfplumber
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables
text_chunks = []
vectorizer = None
embeddings = None

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = []
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text.append(page.extract_text())
    return text

# Function to extract tables from PDF
def extract_tables_from_pdf(pdf_path, page_number):
    with pdfplumber.open(pdf_path) as pdf:
        if page_number <= len(pdf.pages):
            page = pdf.pages[page_number - 1]
            tables = page.extract_tables()
            if tables:
                return tables
    return []

# Function to extract page number from query
def extract_page_number(query):
    match = re.search(r'page\s*(\d+)', query.lower())
    if match:
        return int(match.group(1))
    return None

# Route: Home Page
@app.route('/')
def home():
    return render_template('index.html')

# Route: Upload PDF
@app.route('/upload', methods=['POST'])
def upload_pdf():
    global text_chunks, vectorizer, embeddings
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded PDF
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], "uploaded.pdf")
    file.save(file_path)

    # Extract text from PDF
    text_chunks = extract_text_from_pdf(file_path)
    
    # Check if extracted text is empty
    if not text_chunks or all(chunk.strip() == '' for chunk in text_chunks):
        return jsonify({'error': 'No text found in PDF'}), 400

    # Log extracted text for debugging
    print("Extracted text:", text_chunks[:5])  # Show first 5 lines/chunks of text
    
    # Vectorize the text using TF-IDF
    try:
        vectorizer = TfidfVectorizer()
        embeddings = vectorizer.fit_transform(text_chunks)
    except ValueError as e:
        return jsonify({'error': f'TF-IDF vectorization failed: {str(e)}'}), 400

    return jsonify({'message': 'PDF uploaded and processed successfully'})

# Route: Query PDF
@app.route('/query', methods=['POST'])
def query_pdf():
    global text_chunks, vectorizer, embeddings
    if not text_chunks or vectorizer is None or embeddings is None:
        return jsonify({'error': 'No PDF processed yet'}), 400
    
    query = request.json.get('query', '').lower()
    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Check if query requests tabular data
    if "table" in query or "tabular" in query:
        page_number = extract_page_number(query)
        if page_number is None:
            return jsonify({'error': 'No valid page number found in query.'})
        
        try:
            tables = extract_tables_from_pdf(os.path.join(app.config['UPLOAD_FOLDER'], "uploaded.pdf"), page_number)
            if tables:
                return jsonify({'response': tables})
            else:
                return jsonify({'response': 'No tables found on the specified page.'})
        except Exception as e:
            return jsonify({'error': f'Error processing table query: {e}'})

    # Existing TF-IDF query processing
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, embeddings).flatten()
    most_similar_idx = similarities.argmax()
    response = text_chunks[most_similar_idx]

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
