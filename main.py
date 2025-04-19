import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from numba import njit
import sentencepiece
import orjson as json

import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field
from typing import List
import os

__labels__ = [
    1, # Is toxic 
    0 # Isn't
]

class Tokenizer(sentencepiece.SentencePieceProcessor):
    def __init__(self, model_path="tokenizer.model"):
        super().__init__()
        self.Load(model_path)  # Load the pre-trained sentencepiece model

@dataclass
class TextDataset(Dataset):
    texts: List[str]
    labels: List[int]
    tokenizer: Tokenizer
    max_len: int = 128

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        ids = self.tokenizer.EncodeAsIds(text)
        padded = padd(
            ids,
            max_len=self.max_len,
            text=text
        )
        
        return (
            padded,
            torch.tensor(self.labels[idx], dtype=torch.float32)
        )

def padd(ids, max_len, text, return_tensor=True):
    ids = ids[:max_len]
    padded = np.zeros(max_len, dtype=np.int64)

    length = min(len(ids), max_len)
    padded[:length] = ids
    
    if return_tensor:
        padded = torch.from_numpy(padded)

    return padded

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, batch_first=True)
        self.classifier = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        return self.sigmoid(self.classifier(hidden[-1]))

def train():  
    # Dataset from https://github.com/surge-ai/toxicity/
    df = pd.read_csv('data/dataset.csv')

    # Prepare the corpus file
    input = 'data/corpus.txt'
    with open(input, 'w', encoding='utf-8') as file:
        file.write('\n'.join(df['text'].dropna().astype(str).tolist()))

    # Train the tokenizer
    sentencepiece.SentencePieceTrainer.Train(
        input=input,
        model_prefix='tokenizer',
        vocab_size=3000,
        character_coverage=1.0,
        model_type='bpe',
    )

    t = Tokenizer(model_path='tokenizer.model')

    texts = df['text'].tolist()
    labels = [1 if is_toxic == 'Toxic' else 0 for is_toxic in df['is_toxic'].astype(str).tolist()]

    train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2)

    # Train only
    dataset = TextDataset(texts=train_texts, labels=train_labels, tokenizer=t, max_len=128)

    model = Model(vocab_size=t.GetPieceSize())
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    for _ in range(10):
        for inputs, labels in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)

    torch.save('model.pt')

def predict(text):
    t = Tokenizer()
    model = Model(vocab_size=t.GetPieceSize())

    model.load_state_dict('model.pt')
    model.eval()

    padd(text).unsqueeze
