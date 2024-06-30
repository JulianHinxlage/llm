from collections import defaultdict
import json
import numpy as np

class TrieNode:
    def __init__(self):
        self.children = {}
        self.token = None

# a simple byte pair encoder tokenizer 
class Tokenizer:
    def __init__(self):
        self.tokenToText = []
        self.textToToken = []
        self.trie_root = TrieNode()
        self.max_token_length = 0

    def vocabulary_size(self):
        return len(self.tokenToText)

    def replace_pairs(self, tokens, pair, token):
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == pair:
                new_tokens.append(token)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        return new_tokens

    def count_pairs(self, tokens):
        counts = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i+1])
            counts[pair] += 1
        return counts

    def train(self, text, vocabulary_size, min_frequency=2):
        tokens = [int(c) for c in bytes(text, 'utf-8')]
        self.tokenToText = {i : bytes([i]) for i in range(0, 256)}

        while(len(self.tokenToText) < vocabulary_size):
            counts = self.count_pairs(tokens)

            if len(counts) <= 0:
                break

            max_entry = max(counts.items(), key=lambda i:  i[1])
            max_pair = max_entry[0]

            if max_entry[1] <= min_frequency:
                break

            seq = self.tokenToText[max_pair[0]] + self.tokenToText[max_pair[1]]
            tok = len(self.tokenToText)
            self.tokenToText[tok] = seq
 
            tokens = self.replace_pairs(tokens, max_pair, tok)
            print(f"token={len(self.tokenToText)}/{vocabulary_size}")
        
        self.textToToken = {i[1]: i[0] for i in self.tokenToText.items()}
        self.build_trie()

    def build_trie(self):
        self.trie_root = TrieNode()
        self.max_token_length = 0
        for text, token in self.textToToken.items():
            node = self.trie_root
            self.max_token_length = max(self.max_token_length, len(text))
            for byte in text:
                if byte not in node.children:
                    node.children[byte] = TrieNode()
                node = node.children[byte]
            node.token = token

    def save(self, file):
        with open(file, 'w') as json_file:
            json.dump({
                i[0]: list(i[1]) for i in self.tokenToText.items()
            }, json_file)
    
    def load(self, file):
        with open(file, 'r') as json_file:
            self.tokenToText = json.load(json_file)
            self.tokenToText = {
                int(i[0]): bytes(i[1]) for i in self.tokenToText.items()
            }
        self.textToToken = {i[1]: i[0] for i in self.tokenToText.items()}
        self.build_trie()
    
    def encode(self, text):
        text = bytes(text, 'utf-8')
        tokens = []
        i = 0
        while i < len(text):
            node = self.trie_root
            longest_match = None
            for j in range(min(self.max_token_length, len(text) - i)):
                if text[i+j] not in node.children:
                    break
                node = node.children[text[i+j]]
                if node.token is not None:
                    longest_match = (node.token, j+1)
            if longest_match:
                tokens.append(longest_match[0])
                i += longest_match[1]
            else:
                tokens.append(text[i])
                i += 1
        return tokens

    def decode(self, tokens):
        result = bytearray()
        for tok in tokens:
            if tok >= 0 and tok < self.vocabulary_size():
                result.extend(self.tokenToText[tok])
        return result.decode('utf-8', errors='ignore')
    

def load_tokenizer():
    tokenizer = Tokenizer()
    tokenizer.load("data/tokenizer_8k.json")
    return tokenizer
