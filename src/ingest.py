import os
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from pymilvus import MilvusClient

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