import torch
import numpy as np
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

class Dataset_text(torch.utils.data.Dataset):

    def __init__(self, df):

        self.labels = list(df.label)
        self.texts = [tokenizer(text, 
                               padding='max_length', max_length = 256, truncation=True,
                                return_tensors="pt") for text in df['text']]

    def classes(self):
        return self.labels

    def __len__(self):
        return len(self.labels)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.texts[idx]

    def __getitem__(self, idx):

        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y

class Dataset_embedding(torch.utils.data.Dataset):

    def __init__(self, df, embeddings):

        self.labels = list(df.label)
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