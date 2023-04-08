import numpy as np
from torch.utils.data import Dataset


class SmoteDataset(Dataset):
    def __init__(self, df, labels, transform):
        # self.df = df.values.reshape(-1, 28, 28, 3).astype(np.uint8)
        self.df = df.values.reshape(-1, 84, 84, 3).astype(np.uint8)
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self.transform(self.df[index]), self.labels[index]