import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import os

class TransliterationDataset(Dataset):
    def __init__(self, data_pairs, src_vocab, tgt_vocab, max_len=30):
        self.data_pairs = data_pairs
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_word, tgt_word = self.data_pairs[idx]
        src_encoded = [self.src_vocab["<sos>"]] + [self.src_vocab.get(ch, self.src_vocab["<unk>"]) for ch in src_word.lower()] + [self.src_vocab["<eos>"]]
        tgt_encoded = [self.tgt_vocab["<sos>"]] + [self.tgt_vocab.get(ch, self.tgt_vocab["<unk>"]) for ch in tgt_word] + [self.tgt_vocab["<eos>"]]

        src_encoded = src_encoded[:self.max_len] + [self.src_vocab["<pad>"]] * (self.max_len - len(src_encoded))
        tgt_encoded = tgt_encoded[:self.max_len] + [self.tgt_vocab["<pad>"]] * (self.max_len - len(tgt_encoded))

        return torch.tensor(src_encoded), torch.tensor(tgt_encoded)

def build_vocab(sequences, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
    counter = Counter()
    for seq in sequences:
        counter.update(seq)
    vocab = {ch: idx for idx, ch in enumerate(specials + sorted(counter))}
    return vocab

def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        return [line.strip().split("\t")[:2] for line in f if line.strip() and "\t" in line]

def load_datasets(data_dir, max_len=30):
    train_pairs = read_tsv(os.path.join(data_dir, "train.tsv"))
    dev_pairs   = read_tsv(os.path.join(data_dir, "dev.tsv"))
    test_pairs  = read_tsv(os.path.join(data_dir, "test.tsv"))

    # Combine all data to build vocab
    src_texts = [p[0] for p in train_pairs + dev_pairs + test_pairs]
    tgt_texts = [p[1] for p in train_pairs + dev_pairs + test_pairs]

    src_vocab = build_vocab(src_texts)
    tgt_vocab = build_vocab(tgt_texts)

    train_dataset = TransliterationDataset(train_pairs, src_vocab, tgt_vocab, max_len)
    dev_dataset   = TransliterationDataset(dev_pairs, src_vocab, tgt_vocab, max_len)
    test_dataset  = TransliterationDataset(test_pairs, src_vocab, tgt_vocab, max_len)

    return train_dataset, dev_dataset, test_dataset, src_vocab, tgt_vocab
