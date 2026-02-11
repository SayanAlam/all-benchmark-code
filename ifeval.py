import argparse
import json
import os
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

INPUT_DATA = "google-research/instruction_following_eval/data/input_data.jsonl"
OUTPUT_FILE = "google-research/instruction_following_eval/data/ifeval_responses.jsonl"


def run_ifeval(model_name: str, max_model_len: int = 32000, max_tokens: int = 5000) -> None:
    """Run IFEval benchmark for the given model."""
    # Remove existing output file if present
    if os.path.exists(OUTPUT_FILE):
        os.remove(OUTPUT_FILE)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load vLLM model
    llm = LLM(
        model=model_name,
        trust_remote_code=True,
        max_model_len=max_model_len,
    )

    # Sampling parameters
    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=max_tokens,
    )

    # Load dataset
    dataset = load_dataset(
        "json",
        data_files=INPUT_DATA
    )

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for example in dataset["train"]:
            messages = [
                {"role": "user", "content": example["prompt"]}
            ]
            prompt = example["prompt"]

            # qwen3 style model prompt 
            # prompt = tokenizer.apply_chat_template(
            #     messages,
            #     tokenize=False,
            #     add_generation_prompt=True,
            #     enable_thinking=True
            # )

            # Generate with vLLM
            outputs = llm.generate([prompt], sampling_params)
            response = outputs[0].outputs[0].text.strip()
            response = response.split("</think>", 1)[-1].strip()
            # Save prompt + response in JSONL format
            f.write(json.dumps(
                {"prompt": example["prompt"], "response": response},
                ensure_ascii=False
            ) + "\n")

    print(f"Responses saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run IFEval benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-8B",
        help="Model name or path (default: Qwen/Qwen3-8B)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32000,
        help="Maximum model context length.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=5000,
        help="Maximum number of generated tokens.",
    )
    args = parser.parse_args()
    run_ifeval(args.model, max_model_len=args.max_model_len, max_tokens=args.max_tokens)
