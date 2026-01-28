from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_and_chunk(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    return splitter.split_documents(documents)

if __name__ == "__main__":
    chunks = load_and_chunk("data/priyems.pdf")
    print(f"Loaded {len(chunks)} chunks")
