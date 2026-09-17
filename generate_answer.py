import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import time
from google import genai
from dotenv import load_dotenv

def clean_text_for_api(text):
    return text.encode("ascii", errors="ignore").decode("ascii")

load_dotenv()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

client = chromadb.PersistentClient(
    path="./chroma_db",
    settings=Settings(
        chroma_api_impl="chromadb.api.segment.SegmentAPI",
        anonymized_telemetry=False
    )
)
collection = client.get_or_create_collection(name="finance_qa")

model = SentenceTransformer("all-MiniLM-L6-v2")

question = "How do I start investing with very little money?"
question_embedding = model.encode([question]).tolist()

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)

retrieved_answers = results["metadatas"][0]

context_text = ""
for i, meta in enumerate(retrieved_answers):
    clean_question = clean_text_for_api(results['documents'][0][i])
    clean_answer = clean_text_for_api(meta['answer'])
    context_text += f"\nExample {i+1}:\nQuestion: {clean_question}\nAnswer: {clean_answer}\n"

prompt = f"""You are a helpful personal finance assistant. Answer the user's question using ONLY the information in the examples below. If the examples don't contain enough information, say so honestly instead of making things up.

{context_text}

User's question: {question}

Answer:"""

chat = gemini_client.chats.create(model="gemini-3.6-flash")

max_retries = 3
for attempt in range(max_retries):
    try:
        response = chat.send_message(prompt)
        break
    except Exception as e:
        if attempt < max_retries - 1:
            time.sleep(10)
        else:
            raise

print(response.text)