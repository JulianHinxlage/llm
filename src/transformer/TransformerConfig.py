import torch

class TransformerConfig:
    def __init__(self, name):
        self.name = name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocabulary_size = 8196
        self.context_size = 512
        self.shared_embed_weights = True
        self.dropout_rate = 0.0

        if name == "750k":
            self.embeded_size = 64
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 4
            self.head_count = 4
        elif name == "1.5M":
            self.embeded_size = 96
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 6
            self.head_count = 6
        elif name == "3M":
            self.embeded_size = 160
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 6
            self.head_count = 8
        elif name == "8.5M":
            self.embeded_size = 256
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 8
            self.head_count = 8
        elif name == "21M":
            self.embeded_size = 384
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 10
            self.head_count = 8
        elif name == "42M":
            self.embeded_size = 512
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 12
            self.head_count = 8
        elif name == "91M":
            self.embeded_size = 768
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 12
            self.head_count = 12
        elif name == "260M":
            self.embeded_size = 1024
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 20
            self.head_count = 16
        elif name == "719M":
            self.embeded_size = 1280
            self.hidden_size = self.embeded_size * 4
            self.layer_count = 36
            self.head_count = 16
        else:
            assert "name not valid"
