import llm.LLM as LLM
import os
import torch
import time

def generate(file, text, max_tokens=100):
    llm = LLM.LLM()
    if os.path.exists(file):
        llm.load(file)
    else:
        print("file not found")
        return
    print(f"name={llm.name}, file={file}, parameters={llm.parameter_count()}")

    print(text, end="")
    start = time.time()
    for token in llm.generate_sequence(text, max_tokens, True, 1, 20, 0.95, 0.95):
        print(token, end="")

    print("")
    end = time.time()
    print(f"time={end - start:.2f}s, tok/sec={(max_tokens - 2) / (end - start):.1f}")

if __name__ == "__main__":
    torch.random.manual_seed(2)
    generate("models/91M/v1/model.dat", "in physics ", 1000)

