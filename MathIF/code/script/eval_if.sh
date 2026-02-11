# Change to the script directory to ensure correct relative paths
cd "$(dirname "$0")"

# Arguments:
#   $1: original model name (e.g., Qwen/Qwen2.5-3B)
#   $2: max_token value (e.g., 23000)
MODEL_NAME="${1:-Qwen/Qwen2.5-3B}"
MAX_TOKEN="${2:-23000}"

# Convert model name to filesystem-safe tag (replace '/' with '_')
MODEL_TAG=${MODEL_NAME//\//_}

for dataset in gsm8k math500 minerva olympiad aime
do
    for constraint in single double triple 
    do 
        echo ${MODEL_TAG}_${dataset}_${constraint}
        python3 -u ../eval_if.py \
            --data_path ../../data/${dataset}_${constraint}.jsonl \
            --hypothesis_path output/${MODEL_TAG}_${dataset}_${constraint}_t0.3p0.95max${MAX_TOKEN}seedNone.jsonl

        echo ${MODEL_TAG}_${dataset}_${constraint}_noconstraint
        python3 -u ../eval_if.py \
            --data_path ../../data/${dataset}_${constraint}.jsonl \
            --hypothesis_path output/${MODEL_TAG}_${dataset}_${constraint}_t0.3p0.95max${MAX_TOKEN}seedNone_noconstraint.jsonl
    done
done


