from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os
import shutil
import fitz

from dotenv import load_dotenv
load_dotenv()

CHROMA_PATH = "chroma"
DATA_PATH = "data/"


def main():
    generate_data_store()


def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    save_to_chroma(chunks)


def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith(".pdf"):
            with fitz.open(os.path.join(DATA_PATH, filename)) as doc:
                text = ""
                for page in doc:
                    text += page.get_text()
                # Create a Document object per PDF file
                documents.append(Document(page_content=text, metadata={"filename": filename}))
    return documents

# def load_documents():
#     loader = DirectoryLoader(DATA_PATH, glob="*.md")
#     documents = loader.load()
#     return documents


def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks


def save_to_chroma(chunks: list[Document]):
    # Clear out the database first.
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    # Create a new DB from the documents.
    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(openai_api_key=os.environ.get('OPENAI_API_KEY')), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()