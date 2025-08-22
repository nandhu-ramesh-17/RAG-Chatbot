import os
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
import chromadb

os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')

load_dotenv()

def load_path(path):
    """Read all PDF files from a given directory and extract text."""
    documents = []
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"The path {path} does not exist.")
    
    for file in os.listdir(path):
        if file.endswith('.pdf'):
            file_path = os.path.join(path, file)
            
            try:
                with open(file_path,'rb') as f:
                    reader = PdfReader(f)
                    text = ''
                    for page in reader.pages:
                        text += page.extract_text() or ''

                    documents.append({
                        'file_name': file,
                        'text': text
                    })
            
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    return documents

def chunking(text: str, chunk_size: int = 1000, overlap: int = 200):
    """Split text into chunks with specified size and overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def embed_documents(chunks):
    """Embed the text of documents using Google GenAI."""
    client = genai.Client()
    response = client.models.embed_content(model="gemini-embedding-001", contents=chunks)
    return response.embeddings

def store_embeddings(documents):
    """Store embeddings in a ChromaDB collection."""
    client = chromadb.Client()
    collection = client.get_or_create_collection(name="pdf_embeddings")
    
    for doc in documents:
        chunks = chunking(doc['text'])
        embeddings = embed_documents(chunks)
        for i, (chunk,emb) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{doc['file_name']}_chunk_{i}"
            collection.add(documents=[chunk], embeddings=[emb], ids=[chunk_id])
    
    return collection
