# Ingestion---? Perform Just One Time -- Rerun only when any change in document
# 1. load- Loading Document (.doc,.pdf,.csv,.sql,.html) 2. chunks 3. Embed 4. Store

from pathlib import Path
import re
# from pypdf import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client.models import Distance, VectorParams

from app.core.config import get_embeddings, get_qdrant_client, get_settings

STAFF_ONLY = re.compile(
    r"(?:^|\n)\d+\.\s*(?:Internal:|[^\n]*\(staff only\)).*?(?=\n\d+\. |\Z)",
    re.S | re.I,
)


def load(path: Path) -> list[Document]:     # Return List Of Documents # Langchain Class "Documents"->its attributes-> Page_content,Meta_Data,id
    docs = PyPDFLoader(path).load()         # Creating Multiple Objects
    for doc in docs:
        doc.page_content = STAFF_ONLY.sub("", doc.page_content).strip()
    return [doc for doc in docs if doc.page_content]


def ingest(path: str | None = None) -> int:
    settings = get_settings()

    # 1. Load
    file = Path(path or settings.data_path)
    documents = load(file)

    # 2. Chunk
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)   # splitting Chunks
    if not chunks:
        raise ValueError(f"No text found in {file}")

    # 3. Embed: check the model works before the old collection is deleted
    
    embed = get_embeddings(settings)        # Converting chunks into Vectors

    # 4. Store

    client = get_qdrant_client(settings)

    try : 
        if client.collection_exists(collection_name=settings.collection):
            client.delete_collection(collection_name=settings.collection)

            # Table Creation/collection

        client.create_collection(
            collection_name=settings.collection,
            vectors_config=VectorParams(
                size=settings.embed_dim,
                distance=Distance.COSINE
            )
        )

        # Storing chunks in collection vector databse using this model

        vector_store= QdrantVectorStore(
            client= client,
            collection_name= settings.collection,
            embedding=embed
        )
        vector_store.add_documents(documents=chunks)

    finally:
        client.close()

    return len(chunks)

if __name__=="__main__":
    print(ingest(), "chunks indexed!")
