from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from ingest import load_and_chunk

def create_vector_store():
    chunks = load_and_chunk("data/priyems.pdf")

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("vector_index")

if __name__ == "__main__":
    create_vector_store()
    print("Vector store created successfully")

