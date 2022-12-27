import torch
from torch import nn


class AutoRec(nn.Module):
    def __init__(self, num_hidden, num_users, dropout=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_users, num_hidden, bias=True)
        self.decoder = nn.Linear(num_hidden, num_users, bias=True)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.encoder.weight, std=0.01)
        nn.init.normal_(self.decoder.weight, std=0.01)

    def forward(self, input_var):
        hidden = self.dropout(self.encoder(input_var))
        pred = self.decoder(hidden)
        if self.training:
            return pred * torch.sign(input_var)
        else:
            return pred
