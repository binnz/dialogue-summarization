import torch
from torch.utils.data import Dataset


class DialogDataset(Dataset):

    def __init__(self, train_data, tokenizer):
        self.data = train_data
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)
