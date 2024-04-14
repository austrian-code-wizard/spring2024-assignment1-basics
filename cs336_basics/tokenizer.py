import os
import ast
import pickle
import argparse
import regex as re
from collections import defaultdict
from typing import List, Dict, Tuple, Iterable, Iterator

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: List[str] | None = None
):
    if special_tokens is None:
        special_tokens = []

    pretokens = defaultdict(int)
    with open(input_path) as f:
        for t in re.findall(PAT, f.read()):
            pretokens[t] += 1

    vocab = {
        i: token
        for i, token in enumerate(
            [s.encode("utf-8") for s in special_tokens]
            + [bytes([j]) for j in range(256)]
        )
    }

    pretokens = {
        tuple(bytes((i,)) for i in pretoken.encode("utf-8")): v
        for pretoken, v in pretokens.items()
    }

    pairwise_frequencies = defaultdict(int)
    for pretoken in pretokens:
        for i in range(len(pretoken) - 1):
            pair = pretoken[i : i + 2]
            pairwise_frequencies[pair] += pretokens[pretoken]

    merges = []

    while len(vocab) < vocab_size:
        top_tokens = []
        top_freq = -1
        for token, freq in pairwise_frequencies.items():
            if freq == top_freq:
                top_tokens.append(token)
            elif freq > top_freq:
                top_tokens = [token]
                top_freq = freq
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
        pretokens = new_pretokens
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
    args = parser.parse_args()

    tokenizer = Tokenizer.train_from_file(
        args.input_path, args.vocab_size, args.special_tokens
    )
    tokenizer.save(args.output_path)
    print("Tokenizer training completed and saved to", args.output_path)
