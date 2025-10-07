import os
import regex as re
import time
from typing import BinaryIO

import pickle


PAT = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(
        split_special_token, bytes
    ), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize(chunk_str, special_tokens):
    freqs = {}

    special_pattern = (
        re.compile("|".join(re.escape(tok) for tok in special_tokens))
        if special_tokens
        else None
    )
    sub_chunks = special_pattern.split(chunk_str) if special_pattern else [chunk_str]

    for sub_chunk in sub_chunks:
        for match in PAT.finditer(sub_chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("UTF-8"))
            freqs[match_bytes] = freqs.get(match_bytes, 0) + 1

    return freqs


def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    pretoken = {}
    all_freqs = []
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            freqs = pretokenize(chunk, special_tokens)
            for k, v in freqs.items():
                pretoken[k] = pretoken.get(k, 0) + v
                all_freqs.append(freqs)
    #    import pdb; pdb.set_trace()
    vocab = {i: bytes([i]) for i in range(256)}
    next = max(vocab.keys()) + 1
    for special_token in special_tokens:
        vocab[next] = special_token.encode("utf-8")
        next += 1
    post_pretoken = time.time()
    #    print(f"pretokenization took {post_pretoken - start:.4f} seconds")

    results = pretoken
    merged_list = []

    merges_needed = max(0, vocab_size - len(vocab))
    for _ in range(merges_needed):
        step = iterate(vocab, results, merged_list)
        if step is None:  # no pairs left
            break
        vocab, results, merged_list = step

    now = time.time()
    #  print(f"Took {now - start:.4f} seconds")
    post_pretoken = now
    # print("results", results)
    # print("merged", merged_list)
    # print("vocab", [v for _, v in vocab.items()][256:])
    return vocab, merged_list


def merge_tokens(big: tuple[bytes, ...], sub: tuple[bytes, bytes]) -> tuple[bytes, ...]:
    # Merge all non-overlapping occurrences of `sub` inside `big`, left-to-right.
    out = []
    i = 0
    n = len(big)
    while i < n:
        # since sub is a pair, this check is cheap; if you later generalize, guard for len(sub) > 2
        if i + 1 < n and big[i] == sub[0] and big[i + 1] == sub[1]:
            out.append(sub[0] + sub[1])  # concatenate bytes -> merged token
            i += 2
        else:
            out.append(big[i])
            i += 1
    return tuple(out)


def iterate(vocab, pretoken, merged_list):

    import time

    now = time.time()
    tokens = {}
    pair_counts = {}
    for key, value in pretoken.items():
        for i in range(1, len(key)):
            pair = (key[i - 1], key[i])
            if pair in tokens:
                tokens[pair] += (
                    1 * value
                )  # needs to consider if the pair appears more than once
            else:
                tokens[pair] = value

    first = time.time()

    # print(tokens)

    max_value = max(tokens.values())
    winners = [k for k, v in tokens.items() if v == max_value]
    # print("winners", winners)
    winner = max(winners)
    winner = max(tokens.items(), key=lambda x: (x[1], x[0]))[0]
    next_id = max(vocab) + 1
    vocab[next_id] = winner[0] + winner[1]
    merged_list.append(winner)
    second = time.time()

    results = {}
    for key, value in pretoken.items():
        new_key = merge_tokens(key, winner)
        if new_key in results:
            results[new_key] += value
        else:
            results[new_key] = value

    third = time.time()
    # print(
    #     "times",
    #     "total",  f"{third - now:.3f}s",
    #     "first",  f"{first - now:.3f}s",
    #     "second", f"{second - first:.3f}s",
    #     "third",  f"{third - second:.3f}s",
    # )
    return vocab, results, merged_list
