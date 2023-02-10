from torch import nn
from transformers import BertModel


class BertClassifier(nn.Module):

    def __init__(self, n_classes, model_name='bert-base-uncased', dropout_rate=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = self.dropout(pooled_output)
        output = self.linear(output)
        return output