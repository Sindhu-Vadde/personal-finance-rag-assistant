from load_data import get_combined_data

combined_data = get_combined_data()

print("Total rows:", len(combined_data))

from sentence_transformers import SentenceTransformer
import pickle

model = SentenceTransformer("all-MiniLM-L6-v2")

queries = [row["query"] for row in combined_data]

print("Embedding", len(queries), "queries... this will take a few minutes.")

embeddings = model.encode(queries, show_progress_bar=True)

print("Done. Shape:", embeddings.shape)

with open("embeddings.pkl", "wb") as f:
    pickle.dump({"embeddings": embeddings, "data": combined_data}, f)

print("Saved to embeddings.pkl")