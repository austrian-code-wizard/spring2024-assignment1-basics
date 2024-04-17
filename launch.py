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

train_script = """#!/bin/bash
#SBATCH --job-name=run_train
#SBATCH --partition=batch
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem={gb}G
#SBATCH --time={time}
#SBATCH --output=sbatch/{run_name}-train.out
#SBATCH --error=sbatch/{run_name}-train.err
#SBATCH --gres=gpu:1

# Optional: activate a conda environment to use for this job
eval "$(conda shell.bash hook)"
conda activate cs336_basics

# Print current node
echo "Running on $(hostname)"

python3 cs336_basics/trainer.py --run_name {run_name} --train_path {train_path} --val_path {val_path} --tokenizer_path {tokenizer_path} --cosine_cycle_iters {cosine_cycle_iters} --min_learning_rate {min_learning_rate} --num_iters {num_iters} --val_every {val_every} --checkpoint_every {checkpoint_every} --warmup_iters {warmup_iters} --learning_rate {learning_rate} --batch_size {batch_size}
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

    train_parser = subparsers.add_parser("train", help="Launch a training job")
    train_parser.add_argument(
        "--run_name", type=str, required=True, help="Name of the run"
    )
    train_parser.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    train_parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    train_parser.add_argument(
        "--lr", type=float, default=0.0001, help="Learning rate for training"
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
    elif args.command == "train":
        if args.dataset == "tiny":
            train_dataset = "/data/TinyStoriesV2-GPT4-train.bin"
            val_dataset = "/data/TinyStoriesV2-GPT4-val.bin"
            tokenizer_path = "tiny10k"
            GB = 86
            time = "06:00:00"
            tokens = 327680000
            context = 256
            batch_size = args.batch_size
            train_iters = tokens // (batch_size * context)
            min_lr = 1e-12
            warmup_iters = int(train_iters * 0.1)
            cosine_cycle_iters = train_iters - warmup_iters
            val_every = train_iters // 100
            checkpoint_every = train_iters // 100
            learning_rate = args.lr
            run_name = args.run_name
        else:
            raise ValueError("Invalid dataset")
        with open("tmp.sh", "w") as f:
            f.write(
                train_script.format(
                    gb=GB,
                    time=time,
                    run_name=run_name,
                    train_path=train_dataset,
                    val_path=val_dataset,
                    tokenizer_path=tokenizer_path,
                    cosine_cycle_iters=cosine_cycle_iters,
                    min_learning_rate=min_lr,
                    num_iters=train_iters,
                    val_every=val_every,
                    checkpoint_every=checkpoint_every,
                    warmup_iters=warmup_iters,
                    learning_rate=learning_rate,
                    batch_size=batch_size,
                )
            )
        #os.system("sbatch tmp.sh")
        #os.remove("tmp.sh")
    else:
        print("Invalid command")
        exit(1)


if __name__ == "__main__":
    main()
