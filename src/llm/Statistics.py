
class Statistics:
    def __init__(self):
        self.loss = []
        self.accuracy = []
        self.val_loss = []
        self.val_accuracy = []
        self.perplexity = []
        self.val_perplexity = []

        self.total_trained_tokens = 0
    
    def to_dict(self):
        return {
            "total_trained_tokens": self.total_trained_tokens,
            "loss": self.loss,
            "accuracy": self.accuracy,
            "perplexity": self.perplexity,
            "val_loss": self.val_loss,
            "val_accuracy": self.val_accuracy,
            "val_perplexity": self.val_perplexity,
        }
    
    def load_dict(self, data):
        if 'total_trained_tokens' in data:
            self.total_trained_tokens = data['total_trained_tokens']
        if 'loss' in data:
            self.loss = data['loss']
        if 'accuracy' in data:
            self.accuracy = data['accuracy']
        if 'perplexity' in data:
            self.perplexity = data['perplexity']
        if 'val_loss' in data:
            self.val_loss = data['val_loss']
        if 'val_accuracy' in data:
            self.val_accuracy = data['val_accuracy']
        if 'val_perplexity' in data:
            self.val_perplexity = data['val_perplexity']

    def add(self, tokens, loss, accuracy, perplexity):
        self.loss.append((tokens, loss))
        self.accuracy.append((tokens, accuracy))
        self.perplexity.append((tokens, perplexity))
        self.total_trained_tokens += tokens

    def add_val(self, tokens, loss, accuracy, perplexity):
        self.val_loss.append((tokens, loss))
        self.val_accuracy.append((tokens, accuracy))
        self.val_perplexity.append((tokens, perplexity))
