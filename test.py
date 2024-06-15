from tokenizers import BPETokenizer

tokenizer = BPETokenizer()

data = ["I am Cibi", "I am great"]
# print(tokenizer.tokenize(data))
tokenizer.train(data, max_vocab_size = 300)
# print(len(tokenizer.tokens))
print(tokenizer.tokens[256:])
print(tokenizer.tokenize(data))
tokens = tokenizer.encode(data)
print(tokens)
print(tokenizer.decode(tokens))