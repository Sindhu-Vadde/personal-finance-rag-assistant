import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os
import time
import json
from google import genai
from dotenv import load_dotenv
from load_data import clean_text_for_api



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

def answer_question(question):
    question_embedding = model.encode([question]).tolist()

    results = collection.query(
        query_embeddings=question_embedding,
        n_results=3
    )

    retrieved_answers = results["metadatas"][0]
    retrieved_questions = results["documents"][0]

    context_text = ""
    for i, meta in enumerate(retrieved_answers):
        clean_question = clean_text_for_api(retrieved_questions[i])
        clean_answer = clean_text_for_api(meta['answer'])
        context_text += f"\nExample {i+1}:\nQuestion: {clean_question}\nAnswer: {clean_answer}\n"

    prompt = f"""You are a helpful personal finance assistant. Answer the user's question using ONLY the information in the examples below. If the examples don't contain enough information, say so honestly instead of making things up.

{context_text}

User's question: {question}

Answer:"""

    chat = gemini_client.chats.create(model="gemini-3.5-flash-lite")

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = chat.send_message(prompt)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(15)
            else:
                raise

    return {
        "question": question,
        "retrieved_context": context_text,
        "answer": response.text
    }

with open("eval_questions.json", "r") as f:
    eval_questions = json.load(f)

results = []
for i, question in enumerate(eval_questions):
    print(f"Processing question {i+1}/{len(eval_questions)}: {question[:60]}...")
    result = answer_question(question)
    results.append(result)
    time.sleep(5)

with open("eval_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("Done. Saved", len(results), "results to eval_results.json")