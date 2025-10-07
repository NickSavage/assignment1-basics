import re


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens

    def decode(self, ids: list[int]) -> str:
        if ids == []:
            return ""
        tokens = b""
        for id in ids:
            tokens += self.vocab[id]
        result = tokens.decode("utf-8", errors="replace")
        return result

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []

        split = re.findall(r"\s*\S+\s*", text)
        pretoken = []
        tokens = []

        # step 1: convert to bytes
        for word in split:
            byte_list = [bytes([x]) for x in word.encode("utf-8")]
            print(word, byte_list)
            pretoken.append(byte_list)

        # step 2: apply merges

        # step 3: convert to int

        for word in pretoken:
            found = False
            for token in word:
                for k, v in self.vocab.items():
                    if token == v:
                        tokens.append(k)
                        found = True
                        break
                if not found:
                    raise Exception

        return tokens


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
):
    return Tokenizer(vocab, merges, special_tokens)
