import torch
import random
from llm.HyperParamaters import HyperParameters
import os
import struct

class DatasetConfig:
    def __init__(self):
        self.context_size = 256
        self.batch_size = 32

class Dataset:
    def __init__(self, params : HyperParameters, filename, tokenizer, bytes_to_load = 1000000, offset=0, pre_process=False):
        self.params = params
        self.filename = filename
        self.tokenizer = tokenizer

        if pre_process:
            pre_process_file = filename + "." + str(bytes_to_load) + "_" + str(offset) + "_tokens.bin"
            if os.path.exists(pre_process_file):
                self.tokens = self.load_tokens_from_file(pre_process_file)
                print(f"loaded {len(self.tokens)} tokens")
                pass
            else:
                self.load_text_from_file(filename, bytes_to_load, offset)
                self.write_tokens_to_file(pre_process_file, self.tokens)
                print(f"saved pre processed tokens")
        else:
            self.load_text_from_file(filename, bytes_to_load, offset)

    def write_tokens_to_file(self, filename, tokens):
        with open(filename, 'wb') as file:
            for token in tokens:
                file.write(struct.pack('H', token))
            
    def load_tokens_from_file(self, filename):
        with open(filename, 'rb') as file:
            raw_data = file.read()
            num_tokens = len(raw_data) // 2
            tokens = list(struct.unpack('H' * num_tokens, raw_data))
            return tokens

    def load_text_from_file(self, filename, bytes_to_load, offset):
        self.filename = filename
        with open(filename, 'r', encoding='utf-8') as file:
            file.seek(int(offset))
            data = file.read(int(bytes_to_load))
            print(f"loaded {len(data)} bytes of text")
            self.tokens = self.tokenizer.encode(data)
            print(f"encoded {len(self.tokens)} tokens")
        
    def total_batches(self):
        return int(len(self.tokens) / (self.params.sequence_length * self.params.batch_size))

    def tokens_per_batch(self):
        return self.params.batch_size * self.params.sequence_length

    def next_batch(self):
        x = []
        y = []

        for i in range(self.params.batch_size):
            index = random.randint(0, len(self.tokens) - self.params.sequence_length - 1)
            x.append(self.tokens[index : index + self.params.sequence_length])
            y.append(self.tokens[index + 1 : index + 1 + self.params.sequence_length])

        return torch.Tensor(x).to(device=self.params.config.device), torch.Tensor(y).to(device=self.params.config.device)
