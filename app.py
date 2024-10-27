from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import fitz  # PyMuPDF for PDF text extraction
import os
import tempfile

app = Flask(__name__)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight model for local use

# Local storage for chunks and embeddings
document_chunks = []
chunk_embeddings = []

# Helper function to extract text from a PDF
def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text.strip()

# Function to chunk text into manageable pieces
def chunk_text(text, max_chunk_size=300):
    sentences = text.split(". ")
    current_chunk = ""
    chunks = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    # Append the last chunk if not empty
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# Function to add document to the local "database"
def add_document(file_path):
    text = extract_text_from_pdf(file_path)
    chunks = chunk_text(text)

    for chunk in chunks:
        embedding = model.encode(chunk)
        document_chunks.append(chunk)
        chunk_embeddings.append(embedding)

# Route for the index page with file upload form
@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML file

# Route to handle document upload and analysis
@app.route('/ask', methods=['POST'])
def ask():
    user_input = request.form.get('question', '').strip()
    file = request.files.get('file')

    # Check for greetings
    greetings = ['hello', 'hi', 'hey', 'greetings']
    if any(greeting in user_input.lower() for greeting in greetings):
        return "Hello! How can I help you today?"

    # Check for document analysis request
    if "analyze document" in user_input.lower():
        return '''
            <h1>Please upload the document you want to analyze.</h1>
            <form action="/ask" method="post" enctype="multipart/form-data">
                <input type="file" name="file" required>
                <button type="submit">Upload Document</button>
            </form>
        '''

    # If a file is uploaded, add it to the local database
    if file:
        # Save the file in the system's temporary directory
        temp_dir = tempfile.gettempdir()
        file_path = os.path.join(temp_dir, file.filename)
        file.save(file_path)

        # Add the document to the local database
        add_document(file_path)

        # Clean up the temporary file
        os.remove(file_path)

    # Generate embedding for the question
    question_embedding = model.encode(user_input)

    try:
        # Calculate cosine similarity with document chunks
        similarities = cosine_similarity([question_embedding], chunk_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:3]  # Get top 3 relevant chunks

        # Fetch top matching chunks
        response_text = "<h1>Response</h1><ul>"
        for idx in top_indices:
            response_text += f"<li>{document_chunks[idx]} (Score: {similarities[idx]:.2f})</li>"

        response_text += "</ul>"

        # Check if the highest similarity score is below a threshold (e.g., 0.1)
        if similarities[top_indices[0]] < 0.1:
            response_text += "<p>Please add this to knowledge base suggestions.</p>"
        else:
            response_text += "<a href='/'>Ask another question</a>"

        return response_text

    except ValueError:
        # Handle cases where similarity calculation fails
        return "Please add this question to the knowledge base for future reference."

if __name__ == '__main__':
    app.run(debug=True)
