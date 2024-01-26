from langchain.vectorstores.weaviate import Weaviate
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import VectorDBQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
import os
import weaviate
import json

def init_client():
    return weaviate.Client(
        url = os.environ["WEAVIATE_URL"],
        auth_client_secret=weaviate.AuthApiKey(api_key=os.environ["WEAVIATE_API_KEY"]),
        additional_headers = {"X-OpenAI-Api-Key": os.environ["OPENAI_API_KEY"]}
    )

def init_uscode_db(client: weaviate.Client):
    """
    clears the database, loads it with a directory of PDFs, and returns the vector store for LangChain

    Args:
        client: an initialized weaviate.Client
    
    Returns:
        a Weaviate vector store for LangChain
    """
    schema = {
        "class": "USCode",
        "vectorizer": "text2vec-openai",
        "properties": [
            {
                "name": "text",
                "dataType": ["text"]
            },
            {
                "name": "source",
                "dataType": ["text"]
            },
            {
                "name": "page",
                "dataType": ["number"]
            }
        ]
    }
    client.schema.create_class(schema)
    dir = os.getcwd() + "/data/pdf_uscAll@118-34not31/"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    client.batch.configure(batch_size=100)
    with client.batch as batch:
        for i, fname in enumerate(sorted(os.listdir(dir)), start=1):
            print(f'{i}: {fname}')
            if not fname.endswith(".pdf"):
                continue
            loader = PyPDFLoader(dir + fname)
            documents = loader.load_and_split(text_splitter=text_splitter)
            print(f' {len(documents)} documents')
            for j, doc in enumerate(documents, start=1):
                if j % 1000 == 0:
                    print(f'  {len(documents) - j} documents remaining')
                properties = {
                    "text": doc.page_content,
                    "source": fname,
                    "page": doc.metadata["page"]
                }
                batch.add_data_object(properties, "USCode")
    return load_uscode_db(client)

def load_uscode_db(client: weaviate.Client):
    return Weaviate(client=client, index_name="USCode", text_key="text", embedding=OpenAIEmbeddings(), by_text=False, attributes=["source"])

def read_objects(client: weaviate.Client):
    some_objects = client.data_object.get()
    return json.dumps(some_objects)

def lawsearch_bot(question):
    client = init_client()
    wv8 = init_uscode_db(client) if not client.schema.exists("USCode") else load_uscode_db(client)
    chain = VectorDBQAWithSourcesChain.from_chain_type(ChatOpenAI(temperature=0), chain_type="stuff", vectorstore=wv8)
    return chain({"question": question}, return_only_outputs=True)

lawsearch_bot("What is the punishment for mutilating the flag?")