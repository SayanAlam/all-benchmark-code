mkdir -p ./data
for source in gsm8k math500 minerva olympiad aime
do
    for constraint in single double triple
    do
        wget https://huggingface.co/datasets/TingchenFu/MathIF/resolve/main/${source}_${constraint}.jsonl -O data/${source}_${constraint}.jsonl
    done
done