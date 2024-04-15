import os
import argparse
import ast

tokenizer_script = """#!/bin/bash
#SBATCH --job-name=run_tokenizer
#SBATCH --partition=batch-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={gb}G
#SBATCH --time={time}
#SBATCH --output=sbatch/{name}.out
#SBATCH --error=sbatch/{name}.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

# Print current node
echo "Running on $(hostname)"

python3 cs336_basics/tokenizer.py --input_path {input} --output_path {output} --vocab_size {vocab_size} --special_tokens "{special_tokens}" --log_level debug
"""


def main():
    parser = argparse.ArgumentParser(description="Launch jobs")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    tokenizer_parser = subparsers.add_parser("tokenizer", help="Launch a tokenizer job")
    tokenizer_parser.add_argument(
        "--input", type=str, required=True, help="Path to the input file"
    )
    tokenizer_parser.add_argument(
        "--output", type=str, required=True, help="Path to the output file"
    )
    tokenizer_parser.add_argument(
        "--vocab_size", type=str, required=True, help="Size of the vocabulary"
    )
    tokenizer_parser.add_argument(
        "--special_tokens",
        type=ast.literal_eval,
        required=True,
        help="Special tokens to include in the vocabulary",
    )
    tokenizer_parser.add_argument(
        "--time", type=str, default="00:30:00", help="Time limit for the job"
    )
    tokenizer_parser.add_argument(
        "--gb", type=str, default="16", help="Memory limit for the job"
    )
    tokenizer_parser.add_argument(
        "--name", type=str, default="run_tokenizer%j", help="Name of the job"
    )

    args = parser.parse_args()

    if args.command == "tokenizer":
        with open("tmp.sh", "w") as f:
            f.write(
                tokenizer_script.format(
                    input=args.input,
                    output=args.output,
                    vocab_size=str(args.vocab_size),
                    special_tokens=args.special_tokens,
                    time=args.time,
                    gb=args.gb,
                    name=args.name,
                )
            )
        os.system("sbatch tmp.sh")
        os.remove("tmp.sh")
        print("Launched tokenizer job")
    else:
        print("Invalid command")
        exit(1)


if __name__ == "__main__":
    main()
