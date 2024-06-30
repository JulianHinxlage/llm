import llm.LLM as LLM
import llm.Trainer as Trainer
from llm.HyperParamaters import HyperParameters
import os

def evaluate(file):
    trainer = Trainer.Trainer()
    if os.path.exists(file):
        trainer.load(file)
    else:
        print("file not found")

    print(f"name={trainer.llm.name}, file={file}, parameters={trainer.llm.parameter_count()}, batch_size={trainer.params.batch_size}, lr={trainer.params.lr}, total_tokens_trained={trainer.statistics.total_trained_tokens/1000:.0f}k, description='{trainer.params.description}'")
    
    fineweb = True
    data_file = "data/fineweb.txt"
    trainer.load_data(data_file, trainer.llm.config.context_size * trainer.params.batch_size, 1000000)

    loss, logits, lables, gard_norm = trainer.train_batch(False)
    accuracy, perplexity = trainer.calculate_accuracy(logits, lables)
    print(f"loss={loss}")
    print(f"accuracy={accuracy}")
    print(f"perplexity={perplexity}")

if __name__ == "__main__":
    evaluate("models/91M/v1/model.dat")
