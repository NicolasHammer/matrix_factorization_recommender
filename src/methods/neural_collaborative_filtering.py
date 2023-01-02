"""Define the Neural Matrix Factorization model."""
import random

import torch
from torch import nn
from torch.utils.data import Dataset
from torch import Tensor


class NeuMF(nn.Module):
    """The Neural Matrix Factorization recommendation system model."""

    def __init__(
        self, 
        num_factors: int, 
        num_users: int, 
        num_items: int, 
        nums_hiddens: int, 
        **kwargs
    ) -> None:
        super(NeuMF, self).__init__(**kwargs)
        # Embeddings for GMF
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)

        # Embeddings for MLP
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)

        # MLP
        self.linear_layers = nn.ModuleList()
        if nums_hiddens:
            # Initial layer
            self.linear_layers.extend(
                (nn.Linear(num_factors * 2, nums_hiddens[0], bias=True), nn.ReLU())
            )

            # Hidden layers
            for i in range(1, len(nums_hiddens)):
                self.linear_layers.extend(
                    (
                        nn.Linear(nums_hiddens[i - 1], nums_hiddens[i], bias=True),
                        nn.ReLU(),
                    )
                )

        # Final layer
        self.prediction_layer = nn.Sequential(
            nn.Linear(
                num_factors + (nums_hiddens[-1] if nums_hiddens else 0), 1, bias=False
            ),
            nn.Sigmoid(),
        )

    def forward(self, user_id: int, item_id: int) -> Tensor:
        # Generalized Matrix Factorization
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf

        # Multi-layer perceptron
        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)

        passed_value = torch.cat((p_mlp, q_mlp), axis=1)
        for layer in self.linear_layers:
            passed_value = layer(passed_value)

        # Final step
        return self.prediction_layer(torch.cat((gmf, passed_value), axis=1))


# Create custom dataset
class PRDataset(Dataset):
    def __init__(self, users, items, candidates, num_items):
        self.users = users
        self.items = items
        self.cand = candidates
        self.all = set([i for i in range(num_items)])

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        neg_items = list(self.all - set(self.cand[int(self.users[idx])]))
        indices = random.randint(0, len(neg_items) - 1)
        return self.users[idx], self.items[idx], neg_items[indices]
