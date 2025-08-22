import os
from src.ingest import load_path, chunking, embed_documents, store_in_chroma

def test_ingest_pipeline():
    """
    Simple test pipeline:
    - Load PDFs from a folder
    - Chunk
    - Generate embeddings (Gemini)
    - Store in ChromaDB
    """
    # Example folder containing PDFs
    test_folder = "./data"

    print(f"Loading PDFs from {test_folder}...")
    documents = load_path(test_folder)
    print(f"Found {len(documents)} documents.")

    for doc in documents:
        print(f"\nDocument: {doc['file_name']}")
        chunks = chunking(doc['text'], chunk_size=500, overlap=100)
        print(f" - Total chunks: {len(chunks)}")

        embeddings = embed_documents(chunks)
        print(f" - Generated {len(embeddings)} embeddings.")

    # Store in ChromaDB
    store_in_chroma(documents)
    print("\n✅ Test ingestion pipeline completed successfully.")

if __name__ == "__main__":
    test_ingest_pipeline()
