# loads dataset, trains a BPETokenizer + transformer architecture and validates it on val_file
import pandas as pd
from tokenizers import BPETokenizer

DATASET_PATH = "./datasets/twitter_dataset/"

TRAIN_FILE = "twitter_training.csv"
VAL_FILE = "twitter_validation.csv"
TOKENIZER_VOCAB_FILE = "vocab.pkl"

def load_dataset(filename):
    train_df = pd.read_csv(filename, names = ["col1", "col2", "sentiment", "text"], encoding="utf-8")
    train_df = train_df[["text", "sentiment"]]
    return train_df

if __name__ == "__main__":
    train_df = load_dataset(DATASET_PATH + TRAIN_FILE)
    val_df = load_dataset(DATASET_PATH + VAL_FILE)
    print(f"train df: {len(train_df)}")
    print(f"val df: {len(val_df)}")
    
    tokenizer = BPETokenizer()
    tokenizer.load_vocab(TOKENIZER_VOCAB_FILE)
    print(f"Tokenizer loaded successfully - {tokenizer.vocab_size}.")
    print(tokenizer.encode(train_df['text'][0:1]))




