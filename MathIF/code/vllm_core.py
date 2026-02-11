import argparse
from datasets import load_dataset   
from transformers import AutoModelForCausalLM, AutoTokenizer
# from peft import PeftModel
import os
import torch
from vllm import LLM, SamplingParams
import json
import sys
from pathlib import Path 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_name_or_path', type=str, default="Qwen/Qwen3-1.7B")
parser.add_argument('--tokenizer_name_or_path', type=str, default=None)
parser.add_argument("--test_file",type=str,default=None)
parser.add_argument("--temperature",type=float,default=1.0)
parser.add_argument("--top_p",type=float,default=0.95)
parser.add_argument("--max_token",type=int,default=23000)
parser.add_argument("--n_sample",type=int,default=1)
parser.add_argument("--seed",type=int,default=None)
parser.add_argument("--tensor_parallel_size",type=int,default=1)
parser.add_argument("--max_model_len",type=int,default=88200)
parser.add_argument("--no_constraint",action="store_true")
args = parser.parse_args()





def vllm_inference(prompt):
    try:
        num_gpus = torch.cuda.device_count()
        print(f"Available GPUs: {num_gpus}")
        
        # Simplified VLLM initialization
        llm = LLM(
            model=args.model_name_or_path,
            tokenizer=args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path,
            dtype='bfloat16',
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            max_model_len=args.max_model_len
        )
        print('>>>>>> model loaded successfully')
    except Exception as e:
        print(f"Error initializing VLLM: {e}")
        print("Trying with minimal configuration...")
        # Fallback to minimal configuration
        llm = LLM(
            model=args.model_name_or_path,
            trust_remote_code=True,
            max_model_len=args.max_model_len
        )
        print('>>>>>> model loaded with minimal config')

    sampling_params = SamplingParams(temperature = args.temperature, top_p = args.top_p, max_tokens = args.max_token, seed = args.seed, n = args.n_sample)    
    outputs = llm.generate(prompt, sampling_params)
    sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))
    print('>>>>>> generation done')

    return sorted_outputs



ds = load_dataset('json' if 'json' in args.test_file else 'parquet', data_files=args.test_file, split='train')
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path if args.tokenizer_name_or_path else args.model_name_or_path)
prompt = []
cot = "Let's think step by step and output the final answer within \\boxed{}."
for i in range(len(ds)):
    if 'question' in ds.column_names:
        prompt.append(tokenizer.apply_chat_template([{
            "role": "user",
            "content": ds[i]['question']+ cot + (" ".join(ds[i]["constraint_desc"]) if not args.no_constraint else ""),
        }],tokenize=False, add_generation_prompt=True))
    elif "prompt" in ds.column_names:
        prompt.append(tokenizer.apply_chat_template(ds[i]["prompt"], tokenize=False, add_generation_prompt=True))

print(">>>>>>>>>>>>>>>>>>>>>>>>")
print(prompt[0])
print(">>>>>>>>>>>>>>>>>>>>>>>>")



# Output directory: MathIF/code/script/output (cleared once at start of vllm_if.sh)
output_dir = Path(__file__).resolve().parent / "script" / "output"
output_dir.mkdir(parents=True, exist_ok=True)

output_filename = "{}_{}_t{}p{}max{}seed{}.jsonl".format(
    args.model_name_or_path.replace("/", "_"),
    args.test_file.split("/")[-1].split(".")[0],
    args.temperature,
    args.top_p,
    args.max_token,
    args.seed,
)
if args.no_constraint:
    output_filename = output_filename.replace(".jsonl", "_noconstraint.jsonl")
output_path = output_dir / output_filename

output = vllm_inference(prompt)
fout = open(output_path, "w", encoding="utf8")
for i in range(len(prompt)):
    fout.write(json.dumps({"output": [ output[i].outputs[j].text for j in range(args.n_sample)]}, ensure_ascii=False)+'\n')
fout.close()

