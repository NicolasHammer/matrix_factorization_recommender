"""Define the AutoRec recommendation system."""
import torch
from torch import nn


class AutoRec(nn.Module):
    """The AutoRec recommendation system."""

    encoder: nn.Linear
    decoder: nn.Linear
    dropout: nn.Dropout

    def __init__(self, num_hidden: int, num_users: int, dropout: float=0.05):
        super(AutoRec, self).__init__()
        self.encoder = nn.Linear(num_users, num_hidden, bias=True)
        self.decoder = nn.Linear(num_hidden, num_users, bias=True)
        self.dropout = nn.Dropout(dropout)

        nn.init.normal_(self.encoder.weight, std=0.01)
        nn.init.normal_(self.decoder.weight, std=0.01)

    def forward(self, input_tensor: torch.Tensor):
        """The forward pass of the recommendation system.

        The sign of the forward pass of the recommendation system depends on
        whether or not the recommendation system is training or evaluating.
        
        Args:
            input_tensor: The input tensor to the forward pass

        Returns:
            The result of the forward pass of the recommendation system.
        """
        hidden: torch.Tensor = self.dropout(self.encoder(input_tensor))
        pred: torch.Tensor = self.decoder(hidden)

        # Result of forward pass dependes on whether the model is training
        # or evaluating
        return (
            pred * torch.sign(input_tensor)
            if self.training is True
            else pred
        )
