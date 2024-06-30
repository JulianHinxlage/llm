import Tokenizer
import time

def evaluate_tokenizer(tokenizer : Tokenizer.Tokenizer, test_data):
    total_chars = 0
    total_tokens = 0
    vocab_tokens = 0
    fallback_tokens = 0

    total_chars += len(test_data)
    start_encode_time = time.time()
    tokens = tokenizer.encode(test_data)
    end_encode_time = time.time()
    total_tokens += len(tokens)
    
    for token in tokens:
        if token < 256:
            fallback_tokens += 1
        else:
            vocab_tokens += 1
    
    # Verify perfect reconstruction
    start_decode_time = time.time()
    reconstructed = tokenizer.decode(tokens)
    end_decode_time = time.time()
    assert reconstructed == test_data, "Reconstruction mismatch detected!"

    compression_ratio = total_tokens / total_chars
    vocab_utilization = vocab_tokens / total_tokens
    fallback_rate = fallback_tokens / total_tokens

    for k, v in tokenizer.tokenToText.items():
        print(f"k={k}, v={v}")

    print(f"Total Chars: {total_chars}")
    print(f"Total Tokens: {total_tokens}")
    print(f"Vocal Size: {tokenizer.vocabular_size()}")
    print(f"Compression Ratio: {compression_ratio:.4f}")
    print(f"Vocabulary Utilization: {vocab_utilization:.4f}")
    print(f"Fallback Rate: {fallback_rate:.4f}")
    print(f"Encode Execution time: {end_encode_time - start_encode_time} seconds")
    print(f"Decode Execution time: {end_decode_time - start_decode_time} seconds")



def main():
    tokenizer = Tokenizer.load_tokenizer()
    data = open('data/wikitext.txt', 'r', encoding='utf-8').read(1000000)
    evaluate_tokenizer(tokenizer, data)


if __name__ == "__main__":
    main()