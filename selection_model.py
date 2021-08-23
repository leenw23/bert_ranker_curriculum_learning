import torch
import torch.nn.init as init
from torch import nn
from transformers import BertConfig, BertForMaskedLM, BertModel


class BertSelect(nn.Module):
    def __init__(self, bert: BertModel):
        super(BertSelect, self).__init__()
        self.bert = bert
        self.linear = torch.nn.Linear(768, 1, bias=False)

    def forward(self, ids, mask):
        output, _ = self.bert(ids, mask, return_dict=False)
        cls_ = output[:, 0]
        return self.linear(cls_)

    def get_attention(self, ids, mask):
        output = self.bert(ids, mask, return_dict=True, output_attentions=True)
        prediction = self.linear(output["last_hidden_state"][:, 0])
        return prediction, output["attentions"]
