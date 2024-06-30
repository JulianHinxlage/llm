
import Tokenizer
import time

def train_tokenizer():
    tok =  Tokenizer.Tokenizer()
    data = open('data/fineweb.txt', 'r', encoding='utf-8').read(300000)
    start_time = time.time()
    tok.train(data, 8192)
    tok.save("data/tokenizer_8k_fineweb.json")
    end_time = time.time()
    print(f"Train Execution time: {end_time - start_time} seconds")
    print(f"Vocab Size: {tok.vocabulary_size()}")
    return tok

if __name__ == "__main__":
    tok = train_tokenizer()
