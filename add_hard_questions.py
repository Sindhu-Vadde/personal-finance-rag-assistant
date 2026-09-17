import json

with open("eval_questions.json", "r") as f:
    eval_questions = json.load(f)

hard_questions = [
    "What is the best job to do after finishing master of data science?",
    "What is the current AUD to INR exchange rate?",
    "Is investing in sleeping good?",
    "Should I just spend my 1000 dollars on buying water bottle caps?",
    "Did Stranger Things season 5 get good reviews?",
    "What is the current interest rate on savings in Australian banks?",
    "Is it worth spending money on potato chips?",
    "Should I give my $10,000 to my friend who has a reputation of not giving it back?"
]

eval_questions.extend(hard_questions)

with open("eval_questions.json", "w") as f:
    json.dump(eval_questions, f, indent=2)

print("Total questions now:", len(eval_questions))