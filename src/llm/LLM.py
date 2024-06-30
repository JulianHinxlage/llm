from transformer.Transformer import Transformer, TransformerConfig
import tokenizer.Tokenizer
import torch
import os
from pathlib import Path

class LLM:
    def __init__(self, name=None) -> None:
        if name is not None:
            self.init(name)

    def init(self, name):
        self.name = name
        self.tokenizer = tokenizer.Tokenizer.load_tokenizer()
        self.config = TransformerConfig(name)
        assert(self.config.vocabulary_size != self.tokenizer.vocabulary_size())
        self.model = Transformer(self.config)
        self.model = self.model.to(device=self.config.device)

    def parameter_count(self):
        count = 0
        for p in self.model.parameters():
            count += p.numel()
        return count
    
    def apply_temperature(self, logits, temp):
        return logits / temp

    def apply_top_k(self, logits, top_k):
        if top_k > 0:
            values, indices = torch.topk(logits, top_k)
            logits = torch.ones_like(logits) * float('-inf')
            logits = torch.scatter(logits, 0, indices, values)
        return logits

    def apply_top_p(self, logits, top_p):
        if top_p > 0:
            values, indices = torch.sort(logits, descending=True)
            cumulative = torch.cumsum(torch.softmax(values, dim=-1), dim=-1)
            selected = cumulative <= top_p
            selected[0] = True
            values = values.masked_fill_(selected == False, float('-inf'))
            logits = torch.scatter(logits, -1, indices, values)
        return logits

    def apply_recurring_suppression(self, logits, tokens, recurring_suppression):
        if recurring_suppression > 0:
            min_value = logits.min().item()
            count = len(tokens)
            suppression = 1 - (recurring_suppression / ((torch.arange(count, 0, -1, device=self.config.device) / 4) / recurring_suppression + 1))
            tokens = torch.tensor(tokens, device=self.config.device)
            values = (logits[tokens] - min_value) * suppression + min_value
            logits[tokens] = values
        return logits

    def generate_token(self, tokens, token_history, temp=1.0, top_k=0, top_p=0.0, recurring_suppression=0.0, context_limit=1.0):
        self.model.eval()
        with torch.no_grad():
            max = int(self.config.context_size * context_limit)
            if len(tokens) > max:
                tokens = tokens[-max:]
            
            logits = self.model(torch.Tensor([tokens]))[0][-1]
            logits = self.apply_recurring_suppression(logits, token_history, recurring_suppression)
            logits = self.apply_temperature(logits, temp)
            logits = self.apply_top_k(logits, top_k)
            logits = self.apply_top_p(logits, top_p)

            probs = torch.softmax(logits, dim=-1)
            sample = torch.multinomial(probs, 1).item()
            return sample

    def generate_sequence(self, prompt, max_tokens_to_generate, use_caching=False, temp=1.0, top_k=0, top_p=0.0, recurring_suppression=0.0, context_limit=1.0):
        prompt_tokens = self.tokenizer.encode(prompt)
        generated_tokens = []
        self.model.enable_caching(use_caching)
        for i in range(max_tokens_to_generate):
            if use_caching and i > 0:
                input = [generated_tokens[-1]]
            else:
                input = prompt_tokens + generated_tokens
            #if len(prompt_tokens) + len(generated_tokens) == self.config.context_size:
            #    print("<<##end of context reached##>>", end="")
            token = self.generate_token(input, prompt_tokens + generated_tokens, temp, top_k, top_p, recurring_suppression, context_limit)
            generated_tokens.append(token)
            yield self.tokenizer.decode([token])
        self.model.enable_caching(False)

    def save(self, file, additional_state={}):
        state = {
            "name": self.name,
            "model": self.model.state_dict(),
        }
        state.update(additional_state)
        os.makedirs(Path(file).parent, exist_ok=True)
        torch.save(state, file)

    def load(self, file):
        state = torch.load(file)
        if 'name' in state:
            self.init(state['name'])
        if 'model' in state:
            self.model.load_state_dict(state['model'])
        return state
        