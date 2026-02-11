import json

file_path = "./ft-mistral-7b-instruct-v0.2.jsonl"

total = 0
score_1_count = 0

with open(file_path, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data = json.loads(line)
        total += 1
        if data.get("score") == 1:
            score_1_count += 1

if total > 0:
    percentage = (1 - score_1_count / total) * 100
    print(f"Percentage of score 1: {percentage:.2f}%")
else:
    print("No data found in the file.")
