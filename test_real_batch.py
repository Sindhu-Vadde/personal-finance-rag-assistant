import pickle
import chromadb
from chromadb.config import Settings

with open("embeddings.pkl", "rb") as f:
    saved = pickle.load(f)

embeddings = saved["embeddings"]
combined_data = saved["data"]

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        anonymized_telemetry=False
    )
)
collection = client.get_or_create_collection(name="finance_qa")

n = 100

try:
    collection.add(
        ids=[str(i) for i in range(n)],
        embeddings=embeddings[0:n].tolist(),
        documents=[combined_data[i]["query"] for i in range(n)],
        metadatas=[{"answer": combined_data[i]["answer"], "subreddit": combined_data[i]["subreddit"]} for i in range(n)]
    )
    print(f"SUCCESS with {n} rows, count:", collection.count())
except Exception as e:
    print("CAUGHT ERROR:", e)