"""
Master evaluation runner.

This script runs evaluation for generated responses and aggregates
accuracy/results into a single file: result_accuracy.txt at the project root.
"""

import os
import subprocess
import sys
from run_benchmarks import HF_TOKEN
from huggingface_hub import login
import run_benchmarks

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_FILE = os.path.join(ROOT_DIR, "result_accuracy.txt")
GOOGLE_RESEARCH_DIR = os.path.join(ROOT_DIR, "google-research")
MATHIF_DIR = os.path.join(ROOT_DIR, "MathIF")
SORRY_DATA_DIR = os.path.join(ROOT_DIR, "sorry", "FastChat", "data")
SORRY_JUDGMENT_DIR = os.path.join(
    SORRY_DATA_DIR, "data", "sorry_bench", "model_judgment"
)
SAFETY_BENCH_DIR = os.path.join(ROOT_DIR, "SafetyBench")


def run_ifeval_evaluation() -> str:
    """Run IFEval evaluation and return stdout from the evaluation script."""
    print("Running IFEval evaluation...")
    cmd = [
        sys.executable,
        "-m",
        "instruction_following_eval.evaluation_main",
        "--input_data=./instruction_following_eval/data/input_data.jsonl",
        "--input_response_data=./instruction_following_eval/data/ifeval_responses.jsonl",
        "--output_dir=./instruction_following_eval/data/",
    ]
    result = subprocess.run(
        cmd,
        cwd=GOOGLE_RESEARCH_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print("IFEval eval stdout:", result.stdout)
        if result.stderr:
            print("IFEval eval stderr:", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result.stdout


def run_mathif_evaluation() -> str:
    """Run MathIF evaluation (instruction following + accuracy) and return stdout."""
    print("Running MathIF evaluation...")
    cmd = [
        "bash",
        "code/script/eval_if.sh",
        run_benchmarks.MODEL_NAME,
        str(run_benchmarks.MATHIF_MAX_TOKENS),
    ]
    result = subprocess.run(
        cmd,
        cwd=MATHIF_DIR,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print("MathIF eval stdout:", result.stdout)
        if result.stderr:
            print("MathIF eval stderr:", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result.stdout


def run_sorry_evaluation() -> str:
    """Run Sorry safety evaluation and return the scoring stdout."""
    print("Running Sorry evaluation (safety judge + scoring)...")

    # 1) Delete existing judgment file
    judgment_file = os.path.join(SORRY_JUDGMENT_DIR, "ft-mistral-7b-instruct-v0.2.jsonl")
    if os.path.exists(judgment_file):
        os.remove(judgment_file)

    # 2) Generate new judgments with the safety judge
    cmd_judge = [
        sys.executable,
        "gen_judgment_safety_vllm.py",
        "--bench-name",
        "sorry_bench",
        "--model-list",
        run_benchmarks.SORRY_MODEL_ID,
        "--judge-model",
        "ft-mistral-7b-instruct-v0.2",
    ]
    result_judge = subprocess.run(
        cmd_judge,
        cwd=SORRY_DATA_DIR,
        capture_output=True,
        text=True,
    )
    if result_judge.returncode != 0:
        if result_judge.stdout:
            print("Sorry gen_judgment stdout:", result_judge.stdout)
        if result_judge.stderr:
            print("Sorry gen_judgment stderr:", result_judge.stderr)
        raise subprocess.CalledProcessError(
            result_judge.returncode,
            result_judge.args,
            output=result_judge.stdout,
            stderr=result_judge.stderr,
        )

    # 3) Run scoring script in the judgment directory
    cmd_score = [sys.executable, "score_output.py"]
    result_score = subprocess.run(
        cmd_score,
        cwd=SORRY_JUDGMENT_DIR,
        capture_output=True,
        text=True,
    )
    if result_score.returncode != 0:
        if result_score.stdout:
            print("Sorry score_output stdout:", result_score.stdout)
        if result_score.stderr:
            print("Sorry score_output stderr:", result_score.stderr)
        raise subprocess.CalledProcessError(
            result_score.returncode,
            result_score.args,
            output=result_score.stdout,
            stderr=result_score.stderr,
        )

    return result_score.stdout


def run_safetybench_evaluation() -> str:
    """Run SafetyBench evaluation (eva.py) and return stdout."""
    print("Running SafetyBench evaluation (eva.py)...")
    # Pass model name via environment variable so eva.py can derive the correct file path
    env = os.environ.copy()
    env["MODEL_NAME"] = run_benchmarks.MODEL_NAME
    cmd = [sys.executable, "eva.py"]
    result = subprocess.run(
        cmd,
        cwd=SAFETY_BENCH_DIR,
        env=env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        if result.stdout:
            print("SafetyBench eva stdout:", result.stdout)
        if result.stderr:
            print("SafetyBench eva stderr:", result.stderr)
        raise subprocess.CalledProcessError(
            result.returncode, result.args, output=result.stdout, stderr=result.stderr
        )
    return result.stdout


def main() -> None:
    # Login to Hugging Face using token if provided
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("Logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"Warning: failed to login to Hugging Face Hub: {e}")

    # IFEval (append results, keep previous content)
    ifeval_stdout = run_ifeval_evaluation()
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== IFEval Evaluation ===\n")
        f.write(ifeval_stdout)
        if not ifeval_stdout.endswith("\n"):
            f.write("\n")

    # MathIF
    mathif_stdout = run_mathif_evaluation()
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== MathIF Evaluation ===\n")
        f.write(mathif_stdout)
        if not mathif_stdout.endswith("\n"):
            f.write("\n")

    # Sorry safety evaluation
    sorry_stdout = run_sorry_evaluation()
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== Sorry Evaluation ===\n")
        f.write(sorry_stdout)
        if not sorry_stdout.endswith("\n"):
            f.write("\n")

    # SafetyBench
    safetybench_stdout = run_safetybench_evaluation()
    with open(RESULT_FILE, "a", encoding="utf-8") as f:
        f.write("\n=== SafetyBench Evaluation ===\n")
        f.write(safetybench_stdout)
        if not safetybench_stdout.endswith("\n"):
            f.write("\n")



if __name__ == "__main__":
    main()

