import os
from google import genai
from dotenv import load_dotenv

print("Step A: loading env")
load_dotenv()

print("Step B: creating client")
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

long_prompt = "Explain the following in detail: " + ("investing basics for beginners. " * 300)

print("Prompt length:", len(long_prompt))
print("Step C: calling generate_content")
response = gemini_client.models.generate_content(
    model="gemini-3.6-flash",
    contents=long_prompt
)

print("Step D: got response")
print(response.text[:200])