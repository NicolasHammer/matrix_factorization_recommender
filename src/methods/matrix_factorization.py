import numpy as np
import torch
import torch.nn as nn


class MF(nn.Module):
    def __init__(self, num_factors: int, num_users: int, num_items: int, **kwargs):
        super(MF, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.P.weight, std=0.01)
        nn.init.normal_(self.Q.weight, std=0.01)
        nn.init.normal_(self.user_bias.weight, std=0.01)
        nn.init.normal_(self.item_bias.weight, std=0.01)

    def forward(self, user_id: int, item_id: int):
        P_u = self.P(user_id)
        Q_i = self.Q(item_id)
        b_u = self.user_bias(user_id)
        b_i = self.item_bias(item_id)
        outputs = torch.sum(P_u * Q_i, axis=1) + torch.squeeze(b_u) + torch.squeeze(b_i)
        return outputs
