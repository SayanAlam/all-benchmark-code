# Change to parent directory to avoid naming conflict with 'code' module
cd "$(dirname "$0")/../../.."

# Model name, max_model_len, and max_token: passed from run_benchmarks.py as $1, $2, $3, or use defaults
MODEL_NAME="${1:-Qwen/Qwen2.5-3B}"
MAX_MODEL_LEN="${2:-130000}"
MAX_TOKEN="${3:-23000}"

for model in "$MODEL_NAME"
do
    for dataset in gsm8k math500 minerva olympiad aime
    do
        for constraint in single double triple 
        do
            python3 -u MathIF/code/vllm_core.py \
                --test_file MathIF/data/${dataset}_${constraint}.jsonl \
                --model_name_or_path ${model} \
                --top_p 0.95 \
                --temperature 0.3 \
                --max_token ${MAX_TOKEN} \
                --tensor_parallel_size 1 \
                --max_model_len ${MAX_MODEL_LEN}
            
            python3 -u MathIF/code/vllm_core.py \
                --test_file MathIF/data/${dataset}_${constraint}.jsonl \
                --model_name_or_path ${model}  \
                --top_p 0.95 \
                --temperature 0.3 \
                --max_token ${MAX_TOKEN} \
                --tensor_parallel_size 1 \
                --no_constraint \
                --max_model_len ${MAX_MODEL_LEN}
        done
    done
done

