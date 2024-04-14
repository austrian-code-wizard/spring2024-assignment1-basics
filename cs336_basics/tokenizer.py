import os
import ast
import time
import pickle
import logging
import argparse
import regex as re
from tqdm import tqdm
import multiprocessing
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Iterator

# setup logging
logging.basicConfig(format='%(asctime)s (%(levelname)s): %(message)s')
logger = logging.getLogger(__name__)

# Pretokenization regex
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def count_pretokens(slice):
    result = defaultdict(int)
    for pretoken in re.findall(PAT, slice):
        result[pretoken] += 1
    return result

def train_bpe(
    input_path: str, vocab_size: int, special_tokens: List[str] | None = None
):
    if special_tokens is None:
        special_tokens = []

    start_time = time.time()

    logger.debug("Loading data from disk...")
    with open(input_path) as f:
        data = f.read()
    logger.debug("Took %s seconds to load data from disk", round(time.time() - start_time, 3))

    start_time = time.time()
    pretokens = defaultdict(int)
    if len(data) < 1e7:
        logger.debug("Generating pretokens...")
        pretokens = count_pretokens(data)
    else:
        num_cores = multiprocessing.cpu_count()
        logger.debug(f"Generating pretokens ({num_cores} cores)...")
        data = data.split("\n")
        data_slices = ["\n".join(data[i::num_cores]) for i in range(num_cores)]
        pool = multiprocessing.Pool(num_cores)
        results = pool.map(count_pretokens, data_slices)
        for result in results:
            for t, count in result.items():
                pretokens[t] += count
        pool.close()
        pool.join()
    logger.debug("Took %s seconds to generate pretokens", round(time.time() - start_time, 3))

    start_time = time.time()
    logger.debug("Creating vocab...")
    vocab = {
        i: token
        for i, token in enumerate(
            [s.encode("utf-8") for s in special_tokens]
            + [bytes([j]) for j in range(256)]
        )
    }
    logger.debug("Took %s seconds to create vocab", round(time.time() - start_time, 3))

    start_time = time.time()
    logger.debug("Encoding pretokens...")
    pretokens = {
        tuple(bytes((i,)) for i in pretoken.encode("utf-8")): v
        for pretoken, v in pretokens.items()
    }
    logger.debug("Took %s seconds to encode pretokens", round(time.time() - start_time, 3))

    start_time = time.time()
    logger.debug("Calculating pairwise frequencies...")
    pairwise_frequencies = defaultdict(int)
    for pretoken in pretokens:
        for i in range(len(pretoken) - 1):
            pair = pretoken[i : i + 2]
            pairwise_frequencies[pair] += pretokens[pretoken]
    pairwise_frequencies = {k: v for k, v in sorted(pairwise_frequencies.items(), key=lambda x: x[1])}
    logger.debug(
        "Took %s seconds to calculate pairwise frequencies", round(time.time() - start_time, 3)
    )

    merges = []

    start_time = time.time()
    logger.debug("Training BPE...")
    for _ in tqdm(range(vocab_size - len(vocab))):
        if len(pairwise_frequencies) == 0:
            break

        top_tokens, top_freq = pairwise_frequencies.popitem()
        while len(pairwise_frequencies) > 0:
            token, freq = pairwise_frequencies.popitem()
            if freq < top_freq:
                pairwise_frequencies[token] = freq
                break
            top_token = max(top_tokens, token)

        top_token = max(top_tokens)
        merges.append(top_token)
        top_token_joined = top_token[0] + top_token[1]
        vocab[len(vocab)] = top_token_joined
        new_pretokens = {}
        for pretoken in pretokens:
            cur_idx = 0
            pretoken_count = pretokens[pretoken]
            while cur_idx < len(pretoken) - 1:
                if pretoken[cur_idx : cur_idx + 2] == top_token:

                    # Decrease count if there's a preceding token before the merge pair
                    if cur_idx > 0:
                        pairwise_frequencies[
                            pretoken[cur_idx - 1 : cur_idx + 1]
                        ] -= pretoken_count

                    # Decrease count if there's a next token after the merge pair
                    if cur_idx + 2 < len(pretoken):
                        pairwise_frequencies[
                            pretoken[cur_idx + 1 : cur_idx + 3]
                        ] -= pretoken_count

                    pretoken = (
                        *pretoken[:cur_idx],
                        top_token_joined,
                        *pretoken[cur_idx + 2 :],
                    )

                    if cur_idx > 0:
                        pairwise_frequencies[
                            pretoken[cur_idx - 1 : cur_idx + 1]
                        ] += pretoken_count
                    if cur_idx + 1 < len(pretoken):
                        pairwise_frequencies[
                            pretoken[cur_idx : cur_idx + 2]
                        ] += pretoken_count
                cur_idx += 1
            new_pretokens[pretoken] = pretoken_count
        del pairwise_frequencies[top_token]
        pairwise_frequencies = {
            k: v
            for k, v in sorted(pairwise_frequencies.items(), key=lambda x: x[1])
        }
        pretokens = new_pretokens
    logger.debug("Took %s seconds to train BPE", round(time.time() - start_time, 3))
    return vocab, merges


class Tokenizer:
    def __init__(
        self,
        vocab: Dict[int, bytes],
        merges: List[Tuple[bytes, bytes]],
        special_tokens: List[str] | None = None,
    ) -> None:
        self._vocab = vocab
        self._inv_vocab = {t: i for i, t in self._vocab.items()}
        self._merges = merges
        self._special_tokens = special_tokens if special_tokens else []

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: List[str] | None = None,
    ) -> "Tokenizer":
        return cls(
            pickle.load(open(vocab_filepath, "rb")),
            pickle.load(open(merges_filepath, "rb")),
            special_tokens=special_tokens,
        )

    @classmethod
    def fit(cls, input_path: str, vocab_size: int, special_tokens: List[str]):
        vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def save(self, path: str = ".", prefix: str = "", overwrite: bool = False):
        start_time = time.time()
        os.makedirs(path, exist_ok=True)
        vocab_path = os.path.join(path, prefix + "vocab.pkl")
        if os.path.exists(vocab_path) and not overwrite:
            raise ValueError(
                f"Vocab file {vocab_path} already exists. Set overwrite flag to true if necessary."
            )
        merges_path = os.path.join(path, prefix + "merges.pkl")
        if os.path.exists(merges_path) and not overwrite:
            raise ValueError(
                f"Merge file {merges_path} already exists. Set overwrite flag to true if necessary."
            )
        with open(vocab_path, "wb+") as f:
            pickle.dump(self._vocab, f)
        with open(merges_path, "wb+") as f:
            pickle.dump(self._merges, f)
        logger.debug(
            "Took %s seconds to save tokenizer state to %s", round(time.time() - start_time, 3), path
        )

    @classmethod
    def train_from_file(cls, filepath: str, vocab_size: int, special_tokens: List[str]):
        vocab, merges = train_bpe(filepath, vocab_size, special_tokens)
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> List[int]:
        if self._special_tokens:
            special_token_pattern = re.compile(
                r"("
                + "|".join(
                    map(
                        re.escape,
                        sorted(
                            self._special_tokens, key=lambda x: len(x), reverse=True
                        ),
                    )
                )
                + r")"
            )
            chunks = special_token_pattern.split(text)
        else:
            chunks = [text]
        ids = []
        for chunk in chunks:
            if chunk == "":
                continue
            if chunk in self._special_tokens:
                ids.append(self._inv_vocab[chunk.encode("utf-8")])
                continue
            pretokens = re.findall(PAT, chunk)
            for pretoken in pretokens:
                pretoken = tuple(bytes((i,)) for i in pretoken.encode("utf-8"))
                for pair in self._merges:
                    cur_idx = 0
                    while cur_idx < len(pretoken) - 1:
                        if pretoken[cur_idx : cur_idx + 2] == pair:
                            pretoken = (
                                *pretoken[:cur_idx],
                                pair[0] + pair[1],
                                *pretoken[cur_idx + 2 :],
                            )
                        else:
                            cur_idx += 1
                ids += [self._inv_vocab[token] for token in pretoken]
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for s in iterable:
            for id in self.encode(s):
                yield id

    def decode(self, ids: List[int]) -> str:
        output = b""
        for id in ids:
            output += self._vocab[id]
        return output.decode("utf-8", errors="replace")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tokenizer")
    parser.add_argument("--input_path", type=str, help="Path to the input file")
    parser.add_argument(
        "--output_path", type=str, help="Path to save the tokenizer state"
    )
    parser.add_argument("--vocab_size", type=int, help="Size of the vocabulary")
    parser.add_argument(
        "--special_tokens", type=ast.literal_eval, help="List of special tokens"
    )
    parser.add_argument(
        "--log_level", type=str, default="info", help="Log level (default: info)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", default=True, help="Overwrite the existing tokenizer state"
    )
    args = parser.parse_args()

    logger.setLevel(args.log_level.upper())

    tokenizer = Tokenizer.train_from_file(
        args.input_path, args.vocab_size, args.special_tokens
    )
    tokenizer.save(args.output_path, overwrite=args.overwrite)
    logger.info("Tokenizer training completed and saved to %s", args.output_path)
