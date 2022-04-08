import torch
import numpy as np

class Dataset_embedding(torch.utils.data.Dataset):

    def __init__(self, df, embeddings, target):

        self.labels = list(df[target])
        self.embeddings = embeddings

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_embeddings(self, idx):
        # Fetch a batch of inputs
        return self.embeddings[idx]

    def __getitem__(self, idx):

        batch_embeddings = self.get_batch_embeddings(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_embeddings, batch_y