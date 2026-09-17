import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        anonymized_telemetry=False
    )
)
collection = client.get_or_create_collection(name="finance_qa")

try:
    collection.add(ids=["0"], embeddings=[[0.1]*384], documents=["test"], metadatas=[{"a": "1"}])
    print("SUCCESS")
except Exception as e:
    print("CAUGHT ERROR:", e)
    