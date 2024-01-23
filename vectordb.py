import weaviate
import pdfs

# connect to docker container
client = weaviate.Client(url = 'http://localhost:8080')

# define collection
client.schema.delete_all()
schema = {
    "class": "Document",
    "vectorizer": "text2vec-transformers",
    "properties": [
        {
            "name": "source",
            "dataType": ["text"],
        },
        {
            "name": "title",
            "dataType": ["text"],
        },
        {
            "name": "body",
            "dataType": ["text[]"],
        },
    ]
}
client.schema.create_class(schema)

# get PDFs
docs = pdfs.uscode(True)
client.batch.configure(batch_size=5)
with client.batch as batch:
    for data_object in docs:
        batch.add_data_object(data_object, 'Document')

client.query.get("Document").with_bm25(
    query="a legal document that talks about mutilating the flag"
).with_additional("score").do()