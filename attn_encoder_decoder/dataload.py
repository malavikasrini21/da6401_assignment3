import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import time
import os

# -------------------------------
# Collate function
# -------------------------------
def collate_fn(batch):
    start = time.time()

    batch.sort(key=lambda x: len(x[0]), reverse=True)
    src_batch, tgt_batch = zip(*batch)
    src_lens = [len(seq) for seq in src_batch]
    tgt_lens = [len(seq) for seq in tgt_batch]

    src_padded = pad_sequence(src_batch, padding_value=0, batch_first=True)
    tgt_padded = pad_sequence(tgt_batch, padding_value=0, batch_first=True)

    # print(f"[DEBUG] Collate took {time.time() - start:.4f}s")
    return src_padded, tgt_padded, src_lens, tgt_lens

# -------------------------------
# Character Vocabulary
# -------------------------------
class CharVocab:
    def __init__(self, chars, specials=["<pad>", "<sos>", "<eos>", "<unk>"]):
        self.specials = specials
        self.itos = specials + sorted(set(chars))
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def text2ids(self, text):
        return [self.stoi.get(ch, self.stoi["<unk>"]) for ch in text]

    def ids2text(self, ids):
        return ''.join([
            self.itos[i] for i in ids
            if i not in [self.stoi["<pad>"], self.stoi["<sos>"], self.stoi["<eos>"]]
        ])

# -------------------------------
# Dataset
# -------------------------------
class TransliterationDataset(Dataset):
    def __init__(self, tsv_path, input_vocab, output_vocab):
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.pairs = []

        with open(tsv_path, 'r', encoding='utf-8') as f:
            for line in f:
                tgt, src, freq = line.strip().split('\t')
                src_ids = input_vocab.text2ids(src)
                tgt_ids = [output_vocab.stoi["<sos>"]] + output_vocab.text2ids(tgt) + [output_vocab.stoi["<eos>"]]
                self.pairs.append((
                    torch.tensor(src_ids, dtype=torch.long),
                    torch.tensor(tgt_ids, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

# -------------------------------
# Build vocabulary
# -------------------------------
def build_vocab(file_path, is_input=True):
    chars = set()
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            tgt, src, _ = line.strip().split('\t')
            text = src if is_input else tgt
            chars.update(text)
    return CharVocab(chars)

# -------------------------------
# Dataloader builder
# -------------------------------
def get_dataloaders(data_dir, batch_size=32, pin_memory=True):
    pin_memory = torch.cuda.is_available()
        
    train_path = os.path.join(data_dir, "train.tsv")
    dev_path   = os.path.join(data_dir, "dev.tsv")
    test_path  = os.path.join(data_dir, "test.tsv")
    
    input_vocab = build_vocab(train_path, is_input=True)
    output_vocab = build_vocab(train_path, is_input=False)

    train_ds = TransliterationDataset(train_path, input_vocab, output_vocab)
    dev_ds = TransliterationDataset(dev_path, input_vocab, output_vocab)
    test_ds = TransliterationDataset(test_path, input_vocab, output_vocab)

    num_workers = 4 if torch.cuda.is_available() else 0

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, pin_memory=pin_memory,
                              num_workers=num_workers, persistent_workers=True)
    dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, pin_memory=pin_memory,
                            num_workers=num_workers, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, pin_memory=pin_memory,
                             num_workers=num_workers, persistent_workers=True)

    return train_loader, dev_loader, test_loader, input_vocab, output_vocab
