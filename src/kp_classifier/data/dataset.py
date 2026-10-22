import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def encode_split(tokenizer, df):
    words, chars = tokenizer.encode(df["text"])
    labels = df["label"].to_numpy(dtype=np.int64)
    return TensorDataset(torch.tensor(words), torch.tensor(chars), torch.tensor(labels))


def make_loader(dataset, batch_size, shuffle, seed=0):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator if shuffle else None)
