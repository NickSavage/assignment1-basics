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
        if self.special_tokens is None:
            self.special_tokens = []

    def decode(self, ids: list[int]) -> str:
        if ids == []:
            return ""
        tokens = b""
        for id in ids:
            tokens += self.vocab[id]
        result = tokens.decode("utf-8", errors="replace")
        return result

    def split_preserving_specials(
        self, text: str, special_tokens: list[str]
    ) -> list[str]:
        if special_tokens != []:
            # Escape and sort by length (longest first) so longer specials win in alternation
            escaped = [
                re.escape(t) for t in sorted(set(special_tokens), key=len, reverse=True)
            ]
            specials_pattern = "|".join(escaped)
            pattern = rf"(?:{specials_pattern})|[^\s]+(?:\s+)?"
        else:
            # No empty alternation branch ⇒ no zero-length matches
            pattern = r"[^\s]+(?:\s+)?"

        return re.findall(pattern, text)

    def encode(self, text: str) -> list[int]:
        if text == "":
            return []

        split = self.split_preserving_specials(text, self.special_tokens)
        pretoken = []
        tokens = []

        # step 1: convert to bytes
        for word in split:

            byte_list = [bytes([x]) for x in word.encode("utf-8")]
            pretoken.append(byte_list)

        # step 2: apply merges

        merged = []
        for word in pretoken:

            interim = word
            if len(interim) < 2:
                merged.append(interim)
                continue
            if word in self.special_tokens:
                merged.append(interim)
                continue
            while True:
                new = []
                found = False
                for i in range(len(interim) - 1):
                    c = interim[i]
                    for merge in self.merges:
                        if c == merge[0]:
                            if interim[i + 1] == merge[1]:
                                found = True
                                new.append(c + interim[i + 1])
                                new.extend(interim[i + 2 :])
                                break
                    if found:
                        break
                    else:
                        new.append(c)
                if found:
                    interim = new
                if not found:
                    merged.append(interim)
                    break
        # step 3: convert to int

        for word in merged:
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
