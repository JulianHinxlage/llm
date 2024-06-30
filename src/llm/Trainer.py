import torch
import torch.optim
import llm.LLM as LLM
import llm.Dataset as Dataset
import time
import math
from llm.HyperParamaters import HyperParameters
from llm.Statistics import Statistics

class Trainer:
    def __init__(self, llm : LLM.LLM = None) -> None:
        self.llm = None
        if llm is not None:
            self.init(llm)

    def init(self, params : HyperParameters, llm : LLM.LLM):
        self.llm = llm
        self.params = params
        self.optimizer = torch.optim.AdamW(self.llm.model.parameters(), lr=params.lr)
        self.loss_func = torch.nn.CrossEntropyLoss()
        self.statistics = Statistics()
        self.file = ""

    def scheduler_get_lr(self, iteration):
        lr = 0
        if iteration < self.params.scheduler_warmup_steps:
            lr = self.params.scheduler_max_lr * (iteration+1) / self.params.scheduler_warmup_steps
        else:
            factor = iteration - self.params.scheduler_warmup_steps
            factor /= self.params.scheduler_max_steps - self.params.scheduler_warmup_steps
            if factor > 1.0:
                lr = self.params.scheduler_min_lr
            else:
                factor = (math.cos(factor * 3.1415926) + 1) / 2
                lr = self.params.scheduler_min_lr + factor * (self.params.scheduler_max_lr - self.params.scheduler_min_lr)
        return lr

    def load_data(self, file, bytes=1000000, offset=0):
        self.data = Dataset.Dataset(self.params, file, self.llm.tokenizer, bytes, offset, True)
        self.validation_data = Dataset.Dataset(self.params, file, self.llm.tokenizer, int(bytes * 0.05), offset + bytes, True)

    def save(self, file, update_file_name=True):
        state = {
            "optimizer": self.optimizer.state_dict(),
            "statistics": self.statistics.to_dict(),
            "parameters": self.params.to_dict(),
        }
        self.llm.save(file, state)
        if update_file_name:
            self.file = file

    def load(self, file):
        if self.llm is None:
            llm = LLM.LLM()
            state = llm.load(file)
            params = HyperParameters(llm.config.name, llm.config)
            self.init(params, llm)
        else:
            state = self.llm.load(file)
        
        if 'optimizer' in state:
            self.optimizer.load_state_dict(state['optimizer'])
        if 'statistics' in state:
            self.statistics.load_dict(state['statistics'])
        if 'parameters' in state:
            self.params.load_dict(state['parameters'])
        self.file = file

    def calculate_accuracy(slef, logits, lables):
        with torch.no_grad():
            probs = torch.argmax(logits, dim=-1)
            probs_flat = probs.view(-1)
            lables_flat = lables.view(-1)
            correct_predictions = (probs_flat == lables_flat).sum().item()
            accuracy = correct_predictions / lables_flat.size(0)
            
            probs = torch.softmax(logits, dim=-1).view(-1, logits.size(-1))
            target_probs = probs.gather(dim=-1, index=lables_flat.unsqueeze(-1))
            total_prob = target_probs.log().sum().item()
            perplexity = math.exp(-total_prob / target_probs.numel())
            return accuracy, perplexity

    def validation_batch(self):
        self.llm.model.eval()
        with torch.no_grad():
            input, lables = self.validation_data.next_batch()
            input = input.to(device=self.llm.config.device)
            lables = lables.to(device=self.llm.config.device, dtype=torch.long)
            logits = self.llm.model(input)
            loss = self.loss_func(logits.reshape(-1, logits.size(-1)), lables.view(-1))
            acc, perplexity = self.calculate_accuracy(logits, lables)
            return loss.item(), acc, perplexity

    def train_batch(self, apply_gradient=True, grad_scale=1.0):
        self.llm.model.train()
        input, lables = self.data.next_batch()
        input = input.to(device=self.llm.config.device)
        lables = lables.to(device=self.llm.config.device, dtype=torch.long)
        logits = self.llm.model(input)
        loss = self.loss_func(logits.reshape(-1, logits.size(-1)), lables.view(-1))
        loss.backward()
        loss_value = loss.item()
        grad_norm = 0
        grad_count = 0
        if apply_gradient:
            for param in self.llm.model.parameters():
                if param.grad is not None:
                    param.grad *= grad_scale
                    grad_norm += torch.nn.utils.clip_grad_norm_(param, self.params.max_grad_norm) * param.grad.numel()
                    grad_count += param.grad.numel()
            self.optimizer.step()
            self.optimizer.zero_grad()
            grad_norm /= grad_count
        return loss_value, logits, lables, grad_norm

    def train(self, epochs=1, print_time=2, save_time=10, checkpoint_time=600):
        self.optimizer.zero_grad()
        batches = self.data.total_batches()
        iterations = epochs * batches

        last_print_time = time.time()
        tokens_sice_last_print = 0
        last_save_time = time.time()
        last_checkpoint_time = time.time()

        for iteration in range(iterations):
            batch = iteration % batches
            epoch = iteration // batches

            lr = self.params.lr * self.scheduler_get_lr(self.statistics.total_trained_tokens)
            for params in self.optimizer.param_groups:
                params['lr'] = lr

            batches_per_grad_step = self.params.grad_step_batche_size / self.params.batch_size
            loss, logits, lables, gard_norm = self.train_batch((iteration % batches_per_grad_step) == (batches_per_grad_step-1), 1.0 / batches_per_grad_step)
            acc, perplexity = self.calculate_accuracy(logits, lables)
            token_count = self.data.tokens_per_batch()

            self.statistics.add(token_count, loss, acc, perplexity)
            tokens_sice_last_print += token_count


            t = time.time()
            if t - last_print_time > print_time:
                seconds = t - last_print_time
                tok_per_sec = tokens_sice_last_print/seconds
                eta = ((iterations - iteration) * self.data.tokens_per_batch()) / tok_per_sec

                val_loss, val_acc, val_perplexity = self.validation_batch()
                self.statistics.add_val(tokens_sice_last_print, val_loss, val_acc, val_perplexity)

                print(f"epoch={epoch+1}/{epochs}, batch={batch+1}/{batches}, loss={loss:.5f}, val_loss={val_loss:.5f}, accuracy={acc:.5f}, val_accuracy={val_acc:.5f},\n"
                      + f"perplexity={perplexity:.5f}, val_perplexity={val_perplexity:.5f}, norm={gard_norm:.4f}, lr={lr:.5f}, time={seconds*1000:.0f}ms tok/sec={tok_per_sec:.0f}, total_tokens={self.statistics.total_trained_tokens/1000:.0f}k, eta={eta:.0f}s")
                last_print_time = t
                tokens_sice_last_print = 0

            if t - last_save_time > save_time:
                self.save(self.file)
                last_save_time = t

            if t - last_checkpoint_time > checkpoint_time:
                self.save(self.file + str(time.strftime(".%Y-%m-%d_%H-%M-%S")), False)
                last_checkpoint_time = t
    