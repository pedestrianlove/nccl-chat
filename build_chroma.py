from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import ReadTheDocsLoader
from langchain_chroma import Chroma
from uuid import uuid4

# import the .env file
# from dotenv import load_dotenv
#
# load_dotenv()

# configuration
DATA_PATH = r"rtdocs"
CHROMA_PATH = r"chroma_db"

# initiate the embeddings model
embeddings_model = OllamaEmbeddings(
    base_url="http://192.168.1.12:11434",
    model="nomic-embed-text",
)

# initiate the vector store
vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# loading the PDF document

raw_documents = ReadTheDocsLoader(
    DATA_PATH, encoding="utf-8", custom_html_tag=("article", {"id": "contents"})
).load()

# splitting the document
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

# creating the chunks
chunks = text_splitter.split_documents(raw_documents)
from itertools import batched

for chunk in batched(chunks, 20):
    # creating unique ID's
    uuids = [str(uuid4()) for _ in range(len(chunk))]

    # adding chunks to vector store
    vector_store.add_documents(documents=chunk, ids=uuids)
