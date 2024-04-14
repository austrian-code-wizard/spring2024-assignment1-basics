import os
import argparse


tokenizer_script = """#!/bin/bash
#SBATCH --job-name=run_tokenizer
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={gb}G
#SBATCH --time={time}
#SBATCH --output=run_tokenizer%j.out
#SBATCH --error=run_tokenizer%j.err

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

python3 cs336_basics/tokenizer.py --input_path {input} --output_path {output} --vocab_size {vocab_size} --special_tokens {special_tokens} --log_level debug
"""

def main():
    parser = argparse.ArgumentParser(description="Launch jobs")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    tokenizer_parser = subparsers.add_parser("tokenizer", help="Launch a tokenizer job")
    tokenizer_parser.add_argument("--input", type=str, required=True, help="Path to the input file")
    tokenizer_parser.add_argument("--output", type=str, required=True, help="Path to the output file")
    tokenizer_parser.add_argument("--vocab_size", type=int, required=True, help="Size of the vocabulary")
    tokenizer_parser.add_argument("--special_tokens", type=str, required=True, help="Special tokens to include in the vocabulary")
    tokenizer_parser.add_argument("--time", type=str, default="00:30:00", help="Time limit for the job")
    tokenizer_parser.add_argument("--gb", type=int, default="16", help="Memory limit for the job")
    
    args = parser.parse_args()

    if args.command == "tokenizer":
        with open("tmp.sh", "w") as f:
            f.write(tokenizer_script.format(input=args.input, output=args.output, vocab_size=args.vocab_size, special_tokens=args.special_tokens))
        os.system("sbatch tmp.sh")
        os.remove("tmp.sh")
    else:
        print("Invalid command")
        exit(1)
