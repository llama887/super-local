import glob
import os

import chromadb
import pymupdf
import pymupdf4llm
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

load_dotenv()

openai_client = OpenAI()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = chroma_client.get_or_create_collection(
    name="documents",
    embedding_function=embedding_functions.OpenAIEmbeddingFunction(
        api_key=os.getenv("OPENAI_API_KEY"),
        model_name="text-embedding-3-large",
    ),
)


# Fetch all documents and their metadata
all_docs = chroma_collection.get(include=["metadatas"])  # Fetch only metadata

# Extract unique (url, title) pairs
unique_url_title_pairs = {
    (doc.get("source"), doc.get("title"))
    for doc in tqdm(all_docs["metadatas"], desc="Getting all stored urls and titles")
    if "source" in doc and "title" in doc
}


print("Collection current has:")
for url_title_pair in unique_url_title_pairs:
    print(url_title_pair)

PDF_DIRECTORY = "pdfs"
pdf_paths = glob.glob(os.path.join(PDF_DIRECTORY, "**"), recursive=True)
for pdf_path in pdf_paths:
    if os.path.isfile(pdf_path):
        pdf = pymupdf.open(pdf_path)
        for page_number in range(len(pdf) - 1):
            # https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html#pymupdf4llm-api
            markdown_dict: dict = pymupdf4llm.to_markdown(
                pdf_path,
                pages=[page_number, page_number + 1],
                force_text=True,
                show_progress=True,
                page_chunks=True,
            )
            print(markdown_dict)
            text = markdown_dict["text"]
            metadata = markdown_dict["metadata"]
            print(text)
