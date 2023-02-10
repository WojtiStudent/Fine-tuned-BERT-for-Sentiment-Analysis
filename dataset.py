import torch 
import numpy as np 
from transformers import BertTokenizer
from sklearn.preprocessing import LabelEncoder

# Dictionaries for encoding and decoding labels
labels_encoding = {label: i for i, label in enumerate(["anger", "fear", "joy", "love", "sadness", "surprise"])}
labels_decoding = {i: label for i, label in enumerate(["anger", "fear", "joy", "love", "sadness", "surprise"])}

def encode_labels(df):
    df["label"] = df["label"].map(labels_encoding)
    return df

def decode_labels(df):
    df["label"] = df["label"].map(labels_decoding)
    return df

def create_data_dataloader(df, batch_size, model_name='bert-base-uncased', num_workers=0):
    df = BertDataset(df, model_name)

    return torch.utils.data.DataLoader(df, batch_size=batch_size, num_workers=num_workers)


class BertDataset(torch.utils.data.Dataset):
    def __init__(self, df, model_name='bert-base-uncased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.labels = df["label"].values
        self.max_length = max([len(text.split()) for text in df["text"]])
        self.texts = df["text"].values
    
    def __len__(self):
        return len(self.texts)

    def get_batch_labels(self, idx):
        return torch.tensor(self.labels[idx], dtype=torch.long)

    def get_batch_texts(self, idx):
        text = self.texts[idx]   

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {"text": text,
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten()}
    
    def __getitem__(self, idx):
        return self.get_batch_texts(idx), self.get_batch_labels(idx)