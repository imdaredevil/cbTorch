import pickle as pkl
class Tokenizer:
    def __init__(self):
        self.initialize_vocab()

    def initialize_vocab(self):
        self.vocabulary = {}
        self.tokens = []
        self.vocab_size = 0
    
    def load_vocab(self, file_name):
        self.initialize_vocab()
        with open(file_name, "rb") as f:
            vocab = pkl.load(f)
        self.vocabulary = vocab
        self.tokens = ["" for _ in vocab]
        for key, value in vocab.items():
            self.tokens[value] = key
        self.vocab_size = len(self.tokens)
    
    def save_vocab(self, file_name):
        with open(file_name, "wb") as f:
            pkl.dump(self.vocabulary, f)