import json
import os
import asyncio
from google import genai
from dotenv import load_dotenv
from ragas.llms import llm_factory
from ragas.metrics import Faithfulness
from ragas.dataset_schema import SingleTurnSample

load_dotenv()

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

judge_llm = llm_factory(
    "gemini-3.5-flash-lite",
    provider="google",
    client=client,
    adapter="litellm"
)

with open("eval_results.json", "r") as f:
    eval_results = json.load(f)

faithfulness_metric = Faithfulness(llm=judge_llm)

async def score_all():
    scores = []
    for i, result in enumerate(eval_results):
        print(f"Scoring {i+1}/{len(eval_results)}: {result['question'][:60]}...")

        sample = SingleTurnSample(
            user_input=result["question"],
            response=result["answer"],
            retrieved_contexts=[result["retrieved_context"]]
        )

        try:
            faithfulness_score = await faithfulness_metric.single_turn_ascore(sample)
        except Exception as e:
            faithfulness_score = None
            print("Faithfulness error:", e)

        scores.append({
            "question": result["question"],
            "faithfulness": faithfulness_score
        })

        with open("ragas_scores.json", "w") as f:
            json.dump(scores, f, indent=2)

    return scores

scores = asyncio.run(score_all())

print("Done. Saved scores to ragas_scores.json")
