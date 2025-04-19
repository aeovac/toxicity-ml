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
        ids = self.tokenizer.EncodeAsIds(text)[:self.max_len]
        padded = np.zeros(self.max_len, dtype=np.int64)

        length = min(len(ids), self.max_len)
        padded[:length] = ids[:length]

        inputs = torch.from_numpy(padded)
        texts = torch.tensor(self.labels[idx], dtype=torch.float32)

        return inputs, texts

class Model(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

dataset = TextDataset(texts=texts, labels=labels, tokenizer=t, max_len=128)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = Model()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)