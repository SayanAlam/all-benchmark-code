"""
Master file to run benchmarks. Configure the model name here and call the desired benchmark.
"""
import os
import shutil
import subprocess
import sys
from huggingface_hub import login
from ifeval import run_ifeval

# Project root (directory containing this file)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
MATHIF_DIR = os.path.join(ROOT_DIR, "MathIF")
MATHIF_OUTPUT_DIR = os.path.join(MATHIF_DIR, "code", "script", "output")
SORRY_FASTCHAT_DIR = os.path.join(ROOT_DIR, "sorry", "FastChat")
SORRY_DATA_DIR = os.path.join(SORRY_FASTCHAT_DIR, "data")
SAFETY_BENCH_CODE_DIR = os.path.join(ROOT_DIR, "SafetyBench", "code")
SAFETY_BENCH_DATA_DIR = os.path.join(ROOT_DIR, "SafetyBench", "data")
AIME_DIR = os.path.join(ROOT_DIR, "aime")

HF_TOKEN = "hf token"
# Global configuration
MODEL_NAME = "AzalKhan/Qwen2.5-3B-Instruct_reasoning_instruction_G8_3"
MAX_MODEL_LEN = 32000

# Per-benchmark token limits
IFEVAL_MAX_TOKENS = 5000
SORRY_MODEL_ID = "sorry_response"
SORRY_MAX_NEW_TOKEN = 5000
MATHIF_MAX_TOKENS = 23000
AIME_MAX_TOKENS = 23000


def run_sorry():
    """Run Sorry benchmark (gen_model_answer_vllm.py from sorry/FastChat/data)."""
    print(f"Running Sorry benchmark with model: {MODEL_NAME}, model_id: {SORRY_MODEL_ID}, max_model_len: {MAX_MODEL_LEN}")
    # PYTHONPATH must include sorry/FastChat so "from fastchat..." works when cwd is sorry/FastChat/data
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([SORRY_FASTCHAT_DIR, existing]) if existing else SORRY_FASTCHAT_DIR
    result = subprocess.run(
        [
            sys.executable,
            "gen_model_answer_vllm.py",
            "--bench-name",
            "sorry_bench",
            "--model-path",
            MODEL_NAME,
            "--model-id",
            SORRY_MODEL_ID,
            "--max-new-token",
            str(SORRY_MAX_NEW_TOKEN),
            "--max-model-len",
            str(MAX_MODEL_LEN),
        ],
        cwd=SORRY_DATA_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print("Sorry benchmark stdout:", result.stdout)
        if result.stderr:
            print("Sorry benchmark stderr:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)


def run_safety_bench():
    """Run SafetyBench (evaluate_base.py from SafetyBench/code). Deletes existing output dir before execution."""
    print(f"Running SafetyBench (evaluate_base.py) with model: {MODEL_NAME}...")
    # Delete existing output directory before execution
    if os.path.exists(SAFETY_BENCH_DATA_DIR):
        shutil.rmtree(SAFETY_BENCH_DATA_DIR)
    result = subprocess.run(
        [sys.executable, "evaluate_base.py", "--model", MODEL_NAME],
        cwd=SAFETY_BENCH_CODE_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print("SafetyBench stdout:", result.stdout)
        if result.stderr:
            print("SafetyBench stderr:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)


def run_mathif():
    """Run MathIF generation via bash script (evaluation is handled separately)."""
    print(f"Running MathIF generation with model: {MODEL_NAME}, max_model_len: {MAX_MODEL_LEN}, max_tokens: {MATHIF_MAX_TOKENS}")
    subprocess.run(
        ["bash", "code/script/vllm_if.sh", MODEL_NAME, str(MAX_MODEL_LEN), str(MATHIF_MAX_TOKENS)],
        cwd=MATHIF_DIR,
        check=True,
    )


def run_aime():
    """Run AIME evaluation (aime/aime.py) and save accuracy-only file."""
    print(
        f"Running AIME evaluation with model: {MODEL_NAME}, "
        f"max_model_len: {MAX_MODEL_LEN}, max_new_tokens: {AIME_MAX_TOKENS}"
    )
    result = subprocess.run(
        [
            sys.executable,
            "aime.py",
            "--model",
            MODEL_NAME,
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--max-new-tokens",
            str(AIME_MAX_TOKENS),
        ],
        cwd=AIME_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print("AIME stdout:", result.stdout)
        if result.stderr:
            print("AIME stderr:", result.stderr)
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)


def main():
    # Login to Hugging Face using token if provided
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: failed to login to Hugging Face Hub: {e}")
    
    print(f"Running IFEval benchmark with model: {MODEL_NAME}, max_model_len: {MAX_MODEL_LEN}, max_tokens: {IFEVAL_MAX_TOKENS}")
    run_ifeval(MODEL_NAME, max_model_len=MAX_MODEL_LEN, max_tokens=IFEVAL_MAX_TOKENS)

    print("\nRunning Sorry benchmark...")
    run_sorry()

    print("\nRunning SafetyBench...")
    run_safety_bench()

    print("\nRunning MathIF benchmark...")
    # Delete MathIF output folder once before running (vllm_core.py runs multiple times for different input data)
    if os.path.exists(MATHIF_OUTPUT_DIR):
        shutil.rmtree(MATHIF_OUTPUT_DIR)
    os.makedirs(MATHIF_OUTPUT_DIR, exist_ok=True)
    run_mathif()

    print("\nRunning AIME evaluation...")
    run_aime()


if __name__ == "__main__":
    main()
