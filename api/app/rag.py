from pathlib import Path

from langchain.chat_models import init_chat_model # LLM
from langchain.embeddings import init_embeddings  # Embeddings

from langchain_core.documents import Document # .md, .csv
# from langchain_community.document_loaders import PyPDFLoader # .pdf

from langchain_text_splitters import RecursiveCharacterTextSplitter # chunks
from langchain_core.vectorstores import InMemoryVectorStore # vector Store


#Indexing 

# 1. Load

DATA_DIR = Path(__file__).resolve().parent.parent / "data" # api/data
text = (DATA_DIR / "shop_handbook.md").read_text(encoding="utf-8") # string object

doc = Document(page_content=text, metadata={"source": "shop_handbook.md"})

print(doc)

# 2. Chunk --- 500 to 1000 chunk overlap--- 10 to 20 %

# splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100)
