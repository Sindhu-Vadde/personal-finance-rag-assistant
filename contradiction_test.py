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

def ask_gemini(prompt):
    chat = gemini_client.chats.create(model="gemini-3.5-flash-lite")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat.send_message(prompt)
            return response.text
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                raise

question = "What's a typical average annual return for a diversified stock market index fund?"

question_embedding = model.encode([question]).tolist()
results = collection.query(query_embeddings=question_embedding, n_results=3)

retrieved_answers = results["metadatas"][0]
context_text = ""
for i, meta in enumerate(retrieved_answers):
    clean_question = clean_text_for_api(results['documents'][0][i])
    clean_answer = clean_text_for_api(meta['answer'])
    context_text += f"\nExample {i+1}:\nQuestion: {clean_question}\nAnswer: {clean_answer}\n"

print("ORIGINAL CONTEXT (spot-check the real numbers mentioned):")
print(context_text[:1000])

print("ORIGINAL CONTEXT:")
print(context_text)


corrupted_context = context_text.replace("7-9%", "145-150%").replace("7%", "45%").replace("10%", "50%")
print("\nCORRUPTED CONTEXT:")
print(corrupted_context)

prompt_corrupted = f"""You are a helpful personal finance assistant. Answer the user's question using ONLY the information in the examples below.

{corrupted_context}

User's question: {question}

Answer:"""

print("\n" + "=" * 60)
print("ANSWER USING CORRUPTED (fake) CONTEXT:")
print("=" * 60)
answer = ask_gemini(prompt_corrupted)
print(answer)