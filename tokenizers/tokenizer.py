class Tokenizer:
    def __init__(self):
        self.initialize_vocab()

    def initialize_vocab(self):
        self.vocabulary = {}
        self.tokens = []
        self.vocab_size = 0