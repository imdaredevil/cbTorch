import pandas as pd
from tokenizers import BPETokenizer

DATASET_PATH = "./datasets/twitter_dataset/"

TRAIN_FILE = "twitter_training.csv"
VAL_FILE = "twitter_validation.csv"

def load_dataset(filename):
    train_df = pd.read_csv(filename, names = ["col1", "col2", "sentiment", "text"], encoding="utf-8")
    train_df = train_df[["text", "sentiment"]]
    return train_df

train_df = load_dataset(DATASET_PATH + TRAIN_FILE)
val_df = load_dataset(DATASET_PATH + VAL_FILE)
train_df.dropna(inplace=True)
train_df["text"] = train_df["text"].map(lambda x: x.encode("utf-8").decode("utf-8", "ignore"))

print("training tokenizer")
tokenizer = BPETokenizer(unicode_encoding="utf-8")
tokenizer.train(train_df["text"].tolist(), max_vocab_size = 1000)
print(tokenizer.vocab_size)

