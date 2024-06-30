from transformer.TransformerConfig import TransformerConfig

class HyperParameters:
    def __init__(self, name, config : TransformerConfig):
        self.name = name
        self.config = config
        self.description = ''

        self.max_grad_norm = 1.0
        self.sequence_length = config.context_size
        self.scheduler_warmup_steps = 1 * 1000 * 1000
        self.scheduler_max_steps = 200 * 1000 * 1000
        self.scheduler_min_lr = 0.1
        self.scheduler_max_lr = 1.0

        if name == "750k":
            self.lr = 0.01
            self.batch_size = 48
            self.grad_step_batche_size = self.batch_size
        if name == "1.5M":
            self.lr = 0.01
            self.batch_size = 32
            self.grad_step_batche_size = self.batch_size
        if name == "3M":
            self.lr = 0.005
            self.batch_size = 40
            self.grad_step_batche_size = self.batch_size
        elif name == "8.5M":
            self.lr = 0.0025
            self.batch_size = 32
            self.grad_step_batche_size = self.batch_size
        elif name == "21M":
            self.lr = 0.001
            self.batch_size = 24
            self.grad_step_batche_size = self.batch_size
            self.scheduler_warmup_steps = 5 * 1000 * 1000
        elif name == "42M":
            self.lr = 0.0008
            self.batch_size = 16
            self.grad_step_batche_size = self.batch_size
        elif name == "91M":
            self.lr = 0.0005
            self.batch_size = 12
            self.grad_step_batche_size = self.batch_size
        elif name == "260M":
            self.lr = 0.0004
            self.batch_size = 4
            self.grad_step_batche_size = self.batch_size
        elif name == "719M":
            self.lr = 0.0003
            self.batch_size = 1
            self.grad_step_batche_size = self.batch_size
        else:
            assert True, "name not valid"

    def to_dict(self):
        return {
            "lr": self.lr,
            "batch_size": self.batch_size,
            "grad_step_batche_size": self.grad_step_batche_size,
            "max_grad_norm": self.max_grad_norm,
            "sequence_length": self.sequence_length,
            "scheduler_warmup_steps": self.scheduler_warmup_steps,
            "scheduler_max_steps": self.scheduler_max_steps,
            "scheduler_min_lr": self.scheduler_min_lr,
            "scheduler_max_lr": self.scheduler_max_lr,
            "description": self.description,
        }
    
    def load_dict(self, data):
        if 'lr' in data:
            self.lr = data['lr']
        if 'batch_size' in data:
            self.batch_size = data['batch_size']
        if 'grad_step_batche_size' in data:
            self.grad_step_batche_size = data['grad_step_batche_size']
        if 'max_grad_norm' in data:
            self.max_grad_norm = data['max_grad_norm']
        if 'sequence_length' in data:
            self.sequence_length = data['sequence_length']
        if 'scheduler_warmup_steps' in data:
            self.scheduler_warmup_steps = data['scheduler_warmup_steps']
        if 'scheduler_max_steps' in data:
            self.scheduler_max_steps = data['scheduler_max_steps']
        if 'scheduler_min_lr' in data:
            self.scheduler_min_lr = data['scheduler_min_lr']
        if 'scheduler_max_lr' in data:
            self.scheduler_max_lr = data['scheduler_max_lr']
        if 'description' in data:
            self.description = data['description']