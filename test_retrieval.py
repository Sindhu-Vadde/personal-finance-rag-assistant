import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, util

with open("embeddings.pkl", "rb") as f:
    saved = pickle.load(f)

embeddings = saved["embeddings"]
combined_data = saved["data"]

model = SentenceTransformer("all-MiniLM-L6-v2")

test_question = "How do I get out of credit card debt?"
test_embedding = model.encode([test_question])

similarities = util.cos_sim(test_embedding, embeddings)[0]

best_match_index = np.argmax(similarities)

print("Your question:", test_question)
print("Best match found:", combined_data[best_match_index]["query"])
print("Similarity score:", similarities[best_match_index].item())