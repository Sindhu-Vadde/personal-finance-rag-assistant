from sentence_transformers import SentenceTransformer
model=SentenceTransformer("all-MiniLM-L6-v2")
sentences=["How do I save money on a tight budget?",
           "Tips for cutting expenses",
           "What's the best pizza topping?"
           ]
embeddings=model.encode(sentences)
print(embeddings.shape)

from sentence_transformers import util

similarity_1_2 = util.cos_sim(embeddings[0], embeddings[1])
similarity_1_3 = util.cos_sim(embeddings[0], embeddings[2])

print("Similarity between budget sentences:", similarity_1_2)
print("Similarity between budget and pizza:", similarity_1_3)