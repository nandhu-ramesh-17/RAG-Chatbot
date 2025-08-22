import os
from dotenv import load_dotenv
from pypdf import PdfReader
from google import genai
from pymilvus import MilvusClient, FieldSchema, DataType, CollectionSchema, Collection, connections

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

def chunking(text : str, chunk_size: int = 1000, overlap: int = 200):
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

def store_embeddings(docs, chunks, embeddings):
    """Store document chunks and their embeddings in Milvus."""
    
    # Connect to Milvus
    connections.connect(alias="default", host="localhost", port="19530")
    collection_name = "document_chunks"
    
    # Check if collection exists
    if collection_name in [c.name for c in Collection.list()]:
        collection = Collection(collection_name)
    else:
        # Define schema
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=len(embeddings[0])),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=255)
        ]
        schema = CollectionSchema(fields, description="Document chunks with embeddings")
        collection = Collection(name=collection_name, schema=schema)

    # Prepare data
    sources = [doc["file_name"] for doc in docs for _ in range(len(chunks)//len(docs))]
    entities = [embeddings, chunks, sources]
    
    # Insert and load
    collection.insert(entities)
    collection.load()
    
    return collection

if __name__ == "__main__":
    import os
    
    # Folder containing PDFs
    pdf_folder = "data/"
    
    # Step 1: Load documents
    documents = load_path(pdf_folder)
    if not documents:
        print("No PDF files found in the folder.")
        exit()
    print(f"Loaded {len(documents)} documents.")

    # Step 2: Chunk all documents
    all_chunks = []
    for doc in documents:
        chunks = chunking(doc["text"], chunk_size=1000, overlap=200)
        all_chunks.extend(chunks)
    print(f"Total chunks created: {len(all_chunks)}")

    # Step 3: Embed chunks
    embeddings = embed_documents(all_chunks)
    print(f"Generated embeddings for {len(embeddings)} chunks.")

    # Step 4: Store in Milvus
    collection = store_embeddings(documents, all_chunks, embeddings)
    print(f"Stored chunks in Milvus collection: {collection.name}")
