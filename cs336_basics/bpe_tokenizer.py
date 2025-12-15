import regex as re
from collections.abc import Iterable,Iterator

class Tokenizer():
    def __init__(self, vocab, merges, special_tokens=None):
        None
    
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        None

    def encode(self, text: str) -> list[int]:
        None
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        None
    
    def decode(self, ids: list[int]) -> str:
        None

    def pre_tokenize(str):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        frequency_table = {}
        matches = re.finditer(PAT, str)
        for match in matches:
            byte_string = match.group().encode('utf-8')
            if byte_string in frequency_table:
                frequency_table[byte_string] += 1
            else:
                frequency_table[byte_string] = 1
        return frequency_table
        

    def bpe_tokenizer(input_path:str, vocab_size:int, special_tokens:list[str]):
        if (vocab_size < len(special_tokens) + 256):
            raise ValueError("Vocabulary size much be larger than 256 byte values and special tokens")
        # read from input_path
        data = ""
        # initialize vocabulary with special token <|endoftext|> and the 256 byte values.
        vocab = {i: bytes([i]) for i in range(256)}
        # for i, token in enumerate(["<endoftext>"], start=256):
        #     vocab[i] = token.encode("utf-8")
        # print("Initialized vocabulary")
        # frequency_table = pre_tokenize(str)