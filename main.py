import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
from numba import njit
import sentencepiece
import orjson as json

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import List
import os

TOKENIZER_PATH = "tokenizer.model"
MODEL_PATH = "model.pt"
MAX_LEN = 128
VOCAB_SIZE = 3000
BATCH_SIZE = 32
LR = 0.001
EPOCHS = 10

class Tokenizer(sentencepiece.SentencePieceProcessor):
    def __init__(self, model_path=TOKENIZER_PATH):
        super().__init__()
        self.Load(model_path)  # Load the pre-trained sentencepiece model

@dataclass
class TDataset(Dataset):
    texts: List[str]
    labels: List[int]
    tokenizer: Tokenizer
    max_len: int = 128

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        tokens = self.tokenizer.EncodeAsIds(text)

        if len(tokens) < self.max_len:
            tokens += [0] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return (torch.tensor(tokens), self.labels[idx])

def get_data_loaders(df, tokenizer):
    labels = df["label"]
    counts = np.bincount(labels)
    weights = 1. / counts[labels]

    train_df, test_df = train_test_split(df, test_size=0.2)

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    def collate_fn(batch):
        texts, labels = zip(*batch)

        texts = [torch.tensor(text) for text in texts]
        padded_texts = torch.nn.utils.rnn.pad_sequence(texts, batch_first=True, padding_value=0)
        labels = torch.tensor(labels)
        return padded_texts, labels

    train_loader = DataLoader(
        TDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer),
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(weights[train_df["label"].values], len(train_df)),
        collate_fn=collate_fn
    )

    test_loader = DataLoader(
        TDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer),
        batch_size=BATCH_SIZE,
        sampler=WeightedRandomSampler(weights[test_df["label"].values], len(test_df)),
        collate_fn=collate_fn
    )

    return (train_loader, test_loader)

class Model(nn.Module):
    def __init__(self,vocab_size=VOCAB_SIZE):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.head(hidden[-1]).squeeze(1)

def main():
    # Dataset from https://github.com/surge-ai/toxicity/
    df = pd.read_csv('data/dataset.csv')
    df['label'] = df['is_toxic'].apply(lambda x: 1 if x == 'Toxic' else 0)
    del df['is_toxic']

    if not os.path.exists(TOKENIZER_PATH):    
        # Prepare the corpus file
        input = 'data/corpus.txt'
        with open(input, 'w', encoding='utf-8') as file:
            file.write('\n'.join(df['text'].dropna().astype(str).tolist()))

        sentencepiece.SentencePieceTrainer.Train(
            input=input,
            model_prefix='tokenizer',
            vocab_size=VOCAB_SIZE,
            model_type='bpe',
        )

    t = Tokenizer()

    train_loader, test_loader = get_data_loaders(df, t)

    model = Model(t.GetPieceSize())

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for _ in range(EPOCHS):
        for inputs, labels in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
       
    # Tests
    for inputs, labels in test_loader:
        print(labels.__int__())
        outputs = model(inputs).item()
        
    torch.save(model.state_dict(), MODEL_PATH)

if __name__ == "__main__":
    main() # Entrainement

    t = Tokenizer()
    m = Model(t.GetPieceSize())
    m.load_state_dict(torch.load(MODEL_PATH))

    while True:
        i = input("Your phrase here:")
        tokens = t.EncodeAsIds(i)

        tensor = torch.tensor(tokens).unsqueeze(0)

        with torch.no_grad():
            print(m(tensor).item())
