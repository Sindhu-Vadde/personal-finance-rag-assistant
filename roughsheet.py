from datasets import load_dataset

dataset = load_dataset("Akhil-Theerthala/PersonalFinance-Reddit-QA")
print(dataset["train"][0]["query"])