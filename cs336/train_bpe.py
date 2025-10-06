import os
import regex as re
import time
from typing import BinaryIO

import pickle


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
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    split_text = []
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            chunk = chunk.replace("<|endoftext|>", "")
            split = re.findall(PAT, chunk)

            split_text.extend(split)
    vocab = {i: bytes([i]) for i in range(256)}
    next = max(vocab.keys()) + 1
    for special_token in special_tokens:
        vocab[next] = special_token.encode("utf-8")
        next += 1

    pretoken = {}
    for word in split_text:
        b = word.encode("utf-8")
        tup = tuple(bytes([bt]) for bt in b)  # sequence of 1-byte bytes objects
        pretoken[tup] = pretoken.get(tup, 0) + 1

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


def _check_results(winner, merged_list):
    # Load the snapshot
    snapshot_path = "./tests/_snapshots/test_train_bpe_special_tokens.pkl"
    with open(snapshot_path, "rb") as f:
        expected_data = pickle.load(f)
        i = len(merged_list) - 1
        expected = expected_data["merges"][i]
        if expected != winner:
            print(f"index {i}: expected {expected}, actual {winner}")
            assert expected == winner, ""


def merge_tokens(big: tuple[int], sub: tuple[int]) -> tuple[int]:
    n, m = len(big), len(sub)

    for i in range(n - m + 1):  # sliding window
        if big[i : i + m] == sub:
            new = sub[0] + sub[1]
            # found the subsequence → rebuild with nesting
            return big[:i] + (new,) + big[i + m :]
    return big  # unchanged if not found


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

    # try:
    #     _check_results(winner, merged_list)
    # except AssertionError:

    #     snapshot_path = "./tests/_snapshots/test_train_bpe_special_tokens.pkl"
    #     with open(snapshot_path, "rb") as f:
    #         expected_data = pickle.load(f)
    #         import pdb

    #         pdb.set_trace()
    #         assert False
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
