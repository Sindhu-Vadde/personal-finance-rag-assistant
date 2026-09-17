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

print("Items already in collection:", collection.count())

if collection.count() == 0:
    batch_size = 5000
    total = len(combined_data)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)

        batch_ids = [str(i) for i in range(start, end)]
        batch_embeddings = embeddings[start:end].tolist()
        batch_documents = [row["query"] for row in combined_data[start:end]]
        batch_metadatas = [{"answer": row["answer"], "subreddit": row["subreddit"]} for row in combined_data[start:end]]

        collection.add(
            ids=batch_ids,
            embeddings=batch_embeddings,
            documents=batch_documents,
            metadatas=batch_metadatas
        )

        print(f"Added batch {start} to {end}")

    print("Added", collection.count(), "items to the vector database")
else:
    print("Data already loaded, skipping add step.")