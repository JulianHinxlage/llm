import torch
import torch.nn as nn
import torch.nn.functional as func
import math
from transformer.TransformerConfig import TransformerConfig

class RotaryEncoding:
    def __init__(self, config):
        self.config = config
        self.theta = 10000
        self.freqs = self.precompute(config.embeded_size, config.context_size, self.theta)

    def precompute(self, dim, end, theta = 10000):
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
        t = torch.arange(end, device=freqs.device) 
        freqs = torch.outer(t, freqs).float()
        freqs = torch.polar(torch.ones_like(freqs), freqs)
        return freqs.to(device=self.config.device)

    def extend_precompute(self):
        self.freqs = self.precompute(self.freqs.size(1) * 2, self.freqs.size(0) * 2, self.theta)

    def reshape(self, freqs, x):
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert freqs.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return freqs.view(*shape)

    def apply(self, x, offset=0):
        if x.size(1) + offset > self.freqs.size(0):
            self.extend_precompute()
        complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs = self.reshape(self.freqs[offset:offset+x.shape[1]], complex)
        y = torch.view_as_real(complex * freqs).flatten(3)
        return y.type_as(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, config : TransformerConfig):
        super(MultiHeadAttention, self).__init__()
        self.config = config
        self.attention = nn.Linear(config.embeded_size, config.embeded_size * 3)
        self.projection = nn.Linear(config.embeded_size, config.embeded_size)
        self.cache = None
        self.caching_enabled = False

    def dynamic_slice(self, tensor, dim, start, end):
        slices = [slice(None)] * len(tensor.shape)
        slices[dim] = slice(start, end)
        return tensor[tuple(slices)]

    def update_cache(self, k, v, past_length=0, dim=1):
        if self.cache is None:
            self.cache = (k, v)
        else:
            k_cache, v_cache = self.cache
            if k_cache.shape[dim] > past_length:
                k_cache = self.dynamic_slice(k_cache, dim, -past_length, None)
                v_cache = self.dynamic_slice(v_cache, dim, -past_length, None)
            k = torch.cat((k_cache, k), dim=dim)
            v = torch.cat((v_cache, v), dim=dim)
            self.cache = (k, v)
        return k, v

    def forward(self, x, positional: RotaryEncoding, past_length=0):
        B, T, C = x.size()
        qkv = self.attention(x)
        q,k,v = qkv.split(self.config.embeded_size, dim=2)
        
        if self.caching_enabled:
            k, v = self.update_cache(k, v, past_length)
        
        q = positional.apply(q, past_length)
        k = positional.apply(k)

        q = q.view(B, q.size(1), self.config.head_count, -1).transpose(1, 2)
        k = k.view(B, k.size(1), self.config.head_count, -1).transpose(1, 2)
        v = v.view(B, v.size(1), self.config.head_count, -1).transpose(1, 2)

        use_flash = True
        if use_flash:
            if past_length > 0:
                if q.size(2) > 1:
                    mask = torch.ones(q.size(2), k.size(2), dtype=torch.bool, device=self.config.device).triu(diagonal=1+past_length).logical_not()
                    y = func.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                else:
                    y = func.scaled_dot_product_attention(q, k, v, is_causal=False)
            else:
                y = func.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            k = k.transpose(-2, -1)
            scale = math.sqrt(v.size(-1))
            a = torch.matmul(q, k) / scale
            mask = torch.triu(torch.ones(a.size(), device=self.config.device) * float('-inf'), diagonal=1+past_length)
            a = a + mask
            a = func.softmax(a, dim=-1)
            y = torch.matmul(a, v)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.projection(y)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, config : TransformerConfig):
        super(TransformerBlock, self).__init__()
        self.config = config
        self.attention = MultiHeadAttention(config)
        self.linear1 = nn.Linear(config.embeded_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.embeded_size)
        self.norm1 = nn.LayerNorm(config.embeded_size)
        self.norm2 = nn.LayerNorm(config.embeded_size)
        self.dropout1 = nn.Dropout(config.dropout_rate)
        self.dropout2 = nn.Dropout(config.dropout_rate)

    def forward(self, x, positional: RotaryEncoding, past_length=0):
        residual = x
        x = self.norm1(x)
        x = self.attention(x, positional, past_length)
        x = self.dropout1(x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.linear1(x)
        x = func.relu(x)
        x = self.linear2(x)
        x = self.dropout2(x)
        x = x + residual
        return x

class Transformer(nn.Module):
    def __init__(self, config : TransformerConfig):
        super(Transformer, self).__init__()
        self.config = config
        self.embed_word = nn.Embedding(config.vocabulary_size, config.embeded_size)
        self.embed_positional = RotaryEncoding(config)
        self.blocks = nn.ModuleList([TransformerBlock(config) for i in range(config.layer_count)])
        self.unembed = nn.Linear(config.embeded_size, config.vocabulary_size, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        if config.shared_embed_weights:
            self.unembed.weight = self.embed_word.weight
        self.apply(self.init_weights)
        self.caching_enabled = False
        self.past_length = 0

    def init_weights(self, module):
        std = 0.02
        mean = 0.0
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=mean, std=std)

    def enable_caching(self, enable=True):
        self.caching_enabled = enable
        self.past_length = 0
        for block in self.blocks:
            block.attention.caching_enabled = enable
            if not enable:
                block.attention.cache = None

    def forward(self, x):
        x = x.to(device=self.config.device, dtype=torch.int)
        B, T = x.size()

        x = self.embed_word(x)
        x = self.dropout(x)

        if self.past_length > self.config.context_size - T:
            self.past_length = self.config.context_size - T

        for block in self.blocks:
            x = block(x, self.embed_positional, self.past_length)

        if self.caching_enabled:
            self.past_length += T

        return self.unembed(x)
