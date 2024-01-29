import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI
from pymilvus import utility, connections, Collection

# test connection to db, also needed to use utility functions
connection_args={"uri": os.environ["MILVUS_URI"], "token": os.environ["MILVUS_TOKEN"]}
connections.connect(uri=connection_args["uri"], token=connection_args["token"])

def load_db(collection_name: str):
    return Milvus(
        embedding_function=OpenAIEmbeddings(),
        collection_name=collection_name,
        connection_args=connection_args
    )

def collection_exists(collection_name: str) -> bool:
    if not utility.has_collection(collection_name):
        if collection_name == "USCode":
            init_uscode_db()
            return True
        return False
    return True

def upload_pdfs(directory: str, collection_name: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    db = load_db(collection_name)
    for i, fname in enumerate(sorted(os.listdir(directory)), start=1):
        print(f'{i}: {fname}')
        if not fname.endswith(".pdf"):
            continue
        loader = PyPDFLoader(directory + fname)
        documents = loader.load_and_split(text_splitter=text_splitter)
        num_docs = len(documents)
        print(f" {num_docs} documents")
        [documents[j].metadata.update({"source": fname}) for j in range(num_docs)]
        ids = db.add_documents(documents=documents, embedding=OpenAIEmbeddings(), connection_args=connection_args)
        if num_docs != len(ids):
            print(f" error: expected {num_docs} uploads but got {len(ids)}")

def init_uscode_db():
    directory = os.getcwd() + "/data/pdf_uscAll@118-34not31/"
    collection_name = "USCode"
    upload_pdfs(directory, collection_name)

def delete_file(filename: str, collection_name: str):
    # not atomic i.e. may only delete some then fail (see docs: https://milvus.io/docs/delete_data.md#Delete-Entities)
    if utility.has_collection(collection_name):
        coll = Collection(collection_name)
        coll.load()
        return coll.delete(expr=f"source == '{filename}'")

def query(query: str, collection_name: str):
    if not collection_exists(collection_name):
        print(f"error: {collection_name} not found and can't be initialized")
        return
    
    db = load_db(collection_name)
    relev_docs = db.as_retriever().get_relevant_documents(query)
    for doc in relev_docs:
        print(f'Text: {doc.page_content}')
        print(f'Source: {doc.metadata["source"]}')
        print(f'Page: {doc.metadata["page"] + 1}')
        print()

def qa(question: str, collection_name: str):
    if not collection_exists(collection_name):
        print(f"error: {collection_name} not found and can't be initialized")
        return

    db = load_db(collection_name)
    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever())
    return chain.invoke({"question": question})

print(qa("What is the punishment for mutilating the flag?", "USCode"))