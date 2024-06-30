import llm.LLM as LLM
import llm.Trainer as Trainer
from llm.HyperParamaters import HyperParameters
import os
import torch

def train(file, name, bytes, epochs, description=""):
    trainer = Trainer.Trainer()
    if os.path.exists(file):
        trainer.load(file)
    else:
        llm = LLM.LLM()
        llm.init(name)
        params = HyperParameters(name, llm.config)
        params.description = description
        trainer.init(params, llm)
        trainer.save(file)

    print(f"name={trainer.llm.name}, file={file}, parameters={trainer.llm.parameter_count()}, batch_size={trainer.params.batch_size}/{trainer.params.grad_step_batche_size}, lr={trainer.params.lr}, total_tokens_trained={trainer.statistics.total_trained_tokens/1000:.0f}k, description='{trainer.params.description}'")
    
    data_file = "data/fineweb.txt"
    trainer.load_data(data_file, bytes)

    trainer.train(epochs, 20, 60 * 10, 60 * 60)
    trainer.save(trainer.file)

def test(name):
    llm = LLM.LLM()
    llm.init(name)
    print(f"name={name}, params={llm.parameter_count()}")

def update_model_param(state, key, value):
    for k, v in state.items():
        if k == key:
            state[key] = value
            return
    for k, v in state.items():
        if v is dict:
            update_model_param(v, key, value)

def change_model_param(file, key, value):
    if os.path.exists(file):
        state = torch.load(file)
        update_model_param(state, key, value)
        torch.save(state, file)

if __name__ == "__main__":
    torch.random.manual_seed(1)
    train("models/91M/v1/model.dat", "91M", 1000000000, 1000, "")




