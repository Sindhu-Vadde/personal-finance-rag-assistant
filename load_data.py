from datasets import load_dataset
import re
def clean_text_for_api(text):
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text.encode("ascii", errors="ignore").decode("ascii")

def get_combined_data():
    dataset = load_dataset("Akhil-Theerthala/PersonalFinance-Reddit-QA")
    dataset2 = load_dataset("winddude/reddit_finance_43_250k")
    sample2 = dataset2["train"].shuffle(seed=42).select(range(10000))

    combined_data = []

    for row in dataset["train"]:
        combined_data.append({
            "subreddit": row["subreddit"],
            "query": clean_text_for_api(row["query"]),
            "answer": row["answer"],
            "category": row["category"]
        })

    for row in sample2:
        combined_data.append({
            "subreddit": row["subreddit"],
            "query": clean_text_for_api(row["title"] + " " + row["selftext"]),
            "answer": row["body"],
            "category": None
        })

    return combined_data