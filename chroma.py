import os
import pdfplumber
import chromadb
import textwrap

chroma_client = chromadb.PersistentClient(path="./chroma_db")  # Persistent storage

collection = chroma_client.get_or_create_collection(name="test_collection")

folder_path = "files"

def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF file using pdfplumber."""
    with pdfplumber.open(pdf_path) as pdf:
        text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    return text

def chunk_text(text, chunk_size=500):
    """Splits text into smaller chunks."""
    return textwrap.wrap(text, chunk_size)

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)

    if file_name.endswith(".pdf"):
        text_content = extract_text_from_pdf(file_path)

        if text_content.strip():
            chunks = chunk_text(text_content)


            for idx, chunk in enumerate(chunks):
                collection.add(
                    ids=[f"{file_name}_chunk_{idx}"],
                    documents=[chunk],
                    metadatas=[{"filename": file_name, "chunk": idx}]
                )

print("PDF files added to ChromaDB successfully.")

# Query the database
query_text = ""

results = collection.query(
    query_texts=[query_text],
    n_results=5
)

print(results)
