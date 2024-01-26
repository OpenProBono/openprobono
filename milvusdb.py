import os
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.milvus import Milvus
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import OpenAI
from pymilvus import utility, connections

# test connection to db, also needed to use utility functions
connections.connect(uri=os.environ["MILVUS_URI"], token=os.environ["MILVUS_TOKEN"])

def upload_pdfs(directory: str, collection_name: str):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    embeddings = OpenAIEmbeddings()
    for i, fname in enumerate(sorted(os.listdir(directory)), start=1):
        print(f'{i}: {fname}')
        if not fname.endswith(".pdf"):
            continue
        loader = PyPDFLoader(directory + fname)
        documents = loader.load_and_split(text_splitter=text_splitter)
        print(f' {len(documents)} documents')
        _ = Milvus.from_documents(
            documents,
            embeddings,
            collection_name=collection_name,
            connection_args={
                "uri": os.environ["MILVUS_URI"],
                "token": os.environ["MILVUS_TOKEN"],
                "secure": True,
            }
        )

def init_uscode_db():
    directory = "/Users/njc/Documents/programming/openprobono/data/pdf_uscAll@118-34not31/"
    collection_name = "USCode"
    upload_pdfs(directory, collection_name)

def load_db(collection_name: str):
    return Milvus(
        OpenAIEmbeddings(),
        collection_name=collection_name,
        connection_args={
            "uri": os.environ["MILVUS_URI"],
            "token": os.environ["MILVUS_TOKEN"],
            "secure": True,
        }
    )

def query(query: str, collection_name: str):
    if not utility.has_collection(collection_name):
        if collection_name == "USCode":
            init_uscode_db()
        else:
            return ""
    db = load_db(collection_name)
    relev_docs = db.as_retriever().get_relevant_documents(query)
    for doc in relev_docs:
        _, tail = os.path.split(doc.metadata["source"])
        print(f'Text: {doc.page_content}')
        print(f'Source: {tail}')
        print(f'Page: {doc.metadata["page"] + 1}')
        print()

def qa(question: str, collection_name: str):
    if not utility.has_collection(collection_name):
        if collection_name == "USCode":
            init_uscode_db()
        else:
            return ""
    db = load_db(collection_name)
    chain = RetrievalQAWithSourcesChain.from_chain_type(OpenAI(temperature=0), chain_type="stuff", retriever=db.as_retriever())
    return chain.invoke({"question": question}, return_only_outputs=True)

print(query("What is the punishment for mutilating the flag?", "USCode"))