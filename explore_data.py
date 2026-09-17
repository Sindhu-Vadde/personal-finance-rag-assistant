from datasets import load_dataset
dataset=load_dataset("Akhil-Theerthala/PersonalFinance-Reddit-QA")
print(dataset)
print(dataset["train"][0])

dataset2 = load_dataset("winddude/reddit_finance_43_250k")
print(dataset2)
print(dataset2["train"][0])

row2 = dataset2["train"][0]

combined_query = row2["title"] + " " + row2["selftext"]

print("SUBREDDIT:", row2["subreddit"])
print("QUERY:", combined_query)
print("ANSWER:", row2["body"])
print("CATEGORY:", None)

sample2=dataset2['train'].shuffle(seed=42).select(range(10000))
print(sample2)

combined_data=[]
for row in dataset["train"]:
    combined_data.append({
        "subreddit": row["subreddit"],
        "query": row["query"],
        "answer": row["answer"],
        "category": row["category"]
    })

for row in sample2:
    combined_data.append({
        "subreddit": row["subreddit"],
        "query": row["title"] + " " + row["selftext"],
        "answer": row["body"],
        "category": None
    })

print("Total combined rows:", len(combined_data))
print(combined_data[0])
print(combined_data[-1])