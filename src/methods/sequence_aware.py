"""Define the Caser model."""
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torch import Tensor


class Caser(nn.Module):
    """The Caser model as a recommendation system."""

    P: nn.Embedding
    Q: nn.Embedding

    def __init__(
        self,
        num_factors: int,
        num_users: int,
        num_items: int,
        L: int=5,
        d: int=16,
        d_prime: int=4,
        drop_ratio: float=0.05,
        **kwargs
    ):
        super(Caser, self).__init__(**kwargs)
        self.P = nn.Embedding(num_users, num_factors) 
        self.Q = nn.Embedding(num_items, num_factors)

        # Horizontal convolution layer
        self.d = d
        self.conv_h, self.max_pool = nn.ModuleList(), nn.ModuleList()

        for i in range(1, L + 1):
            self.conv_h.append(
                nn.Conv2d(in_channels=1, out_channels=d, kernel_size=(i, num_factors))
            )
            self.max_pool.append(nn.MaxPool1d(L - i + 1))

        # Vertical convolution layer
        self.d_prime = d_prime
        self.conv_v = nn.Conv2d(in_channels=1, out_channels=d_prime, kernel_size=(L, 1))

        # Fully-connected layer
        self.fc1_dim_v = d_prime * num_factors
        self.fc1_dim_h = d * L

        self.fc = nn.Sequential(
            nn.Linear(self.fc1_dim_v + self.fc1_dim_h, num_factors, bias=True),
            nn.ReLU(),
        )
        self.Q_prime = nn.Embedding(num_items, num_factors * 2)
        self.b = nn.Embedding(num_items, 1)
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, user_id: int, seq, item_id: int) -> Tensor:
        # Construct embeddings
        item_embs = torch.unsqueeze(self.Q(seq), 1)
        user_emb = self.P(user_id)

        # Execute convolutions
        out, out_h, out_v, out_hs = None, None, None, []

        ## Vertical convolutions
        if self.d_prime:
            out_v = self.conv_v(item_embs)
            out_v = torch.reshape(out_v, (out_v.shape[0], self.fc1_dim_v))

        ## Horizontal convolutions
        if self.d:
            for conv, maxp in zip(self.conv_h, self.max_pool):
                conv_out = torch.squeeze(nn.functional.relu(conv(item_embs)), dim=3)
                t = maxp(conv_out)
                pool_out = torch.squeeze(t, dim=2)
                out_hs.append(pool_out)
            out_h = torch.cat((out_hs), dim=1)

        # Fully-connected layer
        out = torch.cat((out_v, out_h), dim=1)
        z = self.fc(self.dropout(out))
        x = torch.cat((z, user_emb), dim=1)
        q_prime_i = torch.squeeze(self.Q_prime(item_id))
        b = torch.squeeze(self.b(item_id))
        res = (x * q_prime_i).sum(1) + b
        return res


class SeqDataset(Dataset):
    def __init__(self, user_ids, item_ids, L, num_users, num_items, candidates):
        user_ids, item_ids = np.array(user_ids), np.array(item_ids)
        sort_idx = np.array(sorted(range(len(user_ids)), key=lambda k: user_ids[k]))
        u_ids, i_ids = user_ids[sort_idx], item_ids[sort_idx]
        temp, u_ids, self.cand = {}, u_ids, candidates
        self.all_items = set([i for i in range(num_items)])
        [temp.setdefault(u_ids[i], []).append(i) for i, _ in enumerate(u_ids)]
        temp = sorted(temp.items(), key=lambda x: x[0])
        u_ids = np.array([i[0] for i in temp])
        idx = np.array([i[1][0] for i in temp])
        self.ns = ns = int(
            sum(
                [
                    c - L if c >= L + 1 else 1
                    for c in np.array([len(i[1]) for i in temp])
                ]
            )
        )
        self.seq_items = np.zeros((ns, L))
        self.seq_users = np.zeros(ns, dtype="int32")
        self.seq_tgt = np.zeros((ns, 1))
        self.test_seq = np.zeros((num_users, L))
        test_users, _uid = np.empty(num_users), None
        for i, (uid, i_seq) in enumerate(self._seq(u_ids, i_ids, idx, L + 1)):
            if uid != _uid:
                self.test_seq[uid][:] = i_seq[-L:]
                test_users[uid], _uid = uid, uid
            self.seq_tgt[i][:] = i_seq[-1:]
            self.seq_items[i][:], self.seq_users[i] = i_seq[:L], uid

    def _win(self, tensor, window_size, step_size=1):
        if len(tensor) - window_size >= 0:
            for i in range(len(tensor), 0, -step_size):
                if i - window_size >= 0:
                    yield tensor[i - window_size : i]
                else:
                    break
        else:
            yield tensor

    def _seq(self, u_ids, i_ids, idx, max_len):
        for i in range(len(idx)):
            stop_idx = None if i >= len(idx) - 1 else int(idx[i + 1])
            for s in self._win(i_ids[int(idx[i]) : stop_idx], max_len):
                yield (int(u_ids[i]), s)

    def __len__(self) -> int:
        return self.ns

    def __getitem__(self, idx):
        neg = list(self.all_items - set(self.cand[int(self.seq_users[idx])]))
        i = random.randint(0, len(neg) - 1)
        return (self.seq_users[idx], self.seq_items[idx], self.seq_tgt[idx], neg[i])
