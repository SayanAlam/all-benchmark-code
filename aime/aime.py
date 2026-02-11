import argparse
import csv
import os
import re
from typing import List, Tuple
import pandas as pd
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    cleaned = text.strip()

    # Remove leading "Answer:" etc.
    cleaned = re.sub(r"^Answer\s*[:\-]\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.rstrip(". ")

    # 🔹 Extract content inside \(\boxed{...}\)
    boxed_match = re.search(r"\\\(\s*\\boxed\{([^}]*)\}\s*\\\)", cleaned)
    if boxed_match:
        cleaned = boxed_match.group(1).strip()
    else:
        # fallback: handle plain \boxed{...} (without \( \))
        boxed_match = re.search(r"\\boxed\{([^}]*)\}", cleaned)
        if boxed_match:
            cleaned = boxed_match.group(1).strip()

    # Remove LaTeX math delimiters like $, \(, \)
    cleaned = re.sub(r"\\[()$]", "", cleaned)
    cleaned = cleaned.strip()

    # If purely numeric, clean up formatting
    digits = re.findall(r"-?\d+", cleaned)
    if digits and len(cleaned) <= 10:
        # Only return number if the box contained just a number
        return digits[-1].lstrip("+")

    return cleaned.lower()



def generate_answer(llm, tokenizer, problem: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant. Solve the following problem."},
        {"role": "user", "content": problem},
    ]

    prompt = problem
    
    # qwen3 style model prompt 
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )


    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=max_new_tokens,
    )

    outputs = llm.generate([text], sampling_params)
    response = outputs[0].outputs[0].text
    return response



def evaluate_on_aime(
    csv_path: str,
    model_name: str,
    max_model_len: int,
    max_new_tokens: int,
) -> Tuple[int, int, float, List[Tuple[str, str, str]]]:
    df = pd.read_csv(csv_path)
    if not {"problem", "answer"}.issubset(df.columns):
        raise ValueError("CSV must contain 'problem' and 'answer' columns")

    llm = LLM(
        model=model_name,
        dtype="auto",
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


    total = 0
    correct = 0
    details: List[Tuple[str, str, str]] = []

    for idx, row in df.iterrows():
        problem = str(row["problem"]) if pd.notna(row["problem"]) else ""
        gold = str(row["answer"]) if pd.notna(row["answer"]) else ""

        pred_raw = generate_answer(llm, tokenizer, problem, max_new_tokens)
        pred_norm = normalize_answer(pred_raw)
        gold_norm = normalize_answer(gold)

        is_correct = pred_norm == gold_norm
        correct += int(is_correct)
        total += 1
        details.append((pred_raw, pred_norm, gold_norm))

        if total % 10 == 0:
            print(f"Processed {total} examples... Accuracy so far: {correct/total:.2%}")

    accuracy = (correct / total) if total > 0 else 0.0
    return correct, total, accuracy, details


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model on AIME dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
        help="Model name or path for evaluation.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32000,
        help="Maximum context length for the model.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=23000,
        help="Maximum number of generated tokens per example.",
    )
    parser.add_argument(
        "--csv-file",
        type=str,
        default="./aime_dataset.csv",
        help="Path to the AIME CSV dataset.",
    )
    parser.add_argument(
        "--accuracy-output",
        type=str,
        default="result_accuracy.txt",
        help="Filename (not path) to save accuracy-only result; it will be saved in the main project directory.",
    )
    args = parser.parse_args()

    # Resolve accuracy output in the main project directory (parent of this file's directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    accuracy_path = os.path.join(project_root, os.path.basename(args.accuracy_output))

    # Remove any existing accuracy file in the project root
    if os.path.exists(accuracy_path):
        os.remove(accuracy_path)

    # Remove any existing predictions file (both relative to dataset dir and current dir)
    legacy_predictions_path = os.path.join(os.path.dirname(args.csv_file), "aime_predictions.csv")
    if os.path.exists(legacy_predictions_path):
        os.remove(legacy_predictions_path)
    if os.path.exists("aime_predictions.csv"):
        os.remove("aime_predictions.csv")

    try:
        correct, total, accuracy, details = evaluate_on_aime(
            csv_path=args.csv_file,
            model_name=args.model,
            max_model_len=args.max_model_len,
            max_new_tokens=args.max_new_tokens,
        )
        print("\nAIME evaluation complete.")
        print(f"Total: {total}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.2%}")

        # Save only accuracy (plus counts) to a separate file in project root
        with open(accuracy_path, "w", encoding="utf-8") as f:
            f.write(f"aime evaluation result\n")
            f.write(f"model: {args.model}\n")
            f.write(f"max_model_len: {args.max_model_len}\n")
            f.write(f"max_new_tokens: {args.max_new_tokens}\n")
            f.write(f"total: {total}\n")
            f.write(f"correct: {correct}\n")
            f.write(f"accuracy: {accuracy:.6f}\n")
        print(f"Saved accuracy summary to {args.accuracy_output}")
    except FileNotFoundError:
        print(f"CSV file not found at {args.csv_file}")
    except Exception as e:
        print(f"Error during evaluation: {e}")