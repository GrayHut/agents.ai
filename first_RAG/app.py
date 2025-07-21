import os
from dotenv import load_dotenv
import chromadb
from openai import OpenAI
from chromadb.utils import embedding_functions

load_dotenv()

openai_key=os.getenv("OPEN_AI_KEY")

openai_ef=embedding_functions.OpenAIEmbeddingFunction(
    api_key=openai_key,
    model_name="text-embedding-3-small"
)

#initialize the chroma client with persistence

chroma_client=chromadb.PersistentClient(path="chroma_persistent_storage")
collection_name="document_qa_collection"
collection=chroma_client.get_or_create_collection(name=collection_name, embedding_function=openai_ef)

client=OpenAI(api_key=openai_key)

resp=client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role":"system", "content": "You are a helpful assistant"},
        {"role":"user", "content": "What is the human life expenctancy in the United States"},
    ],
)

print(resp)