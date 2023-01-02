"""Define the AutoRec recommendation system."""
import torch
from torch import nn
from torch import Tensor


class AutoRec(nn.Module):
    """The AutoRec recommendation system."""

    encoder: nn.Linear
    decoder: nn.Linear
    dropout: nn.Dropout

    def __init__(
        self, num_hidden: int, num_users: int, dropout: float = 0.05
    ) -> None:
        super(AutoRec, self).__init__()

        # Intialize the encoder of the AutoRec model with values from the
        # normal distribution
        self.encoder = nn.Linear(num_users, num_hidden, bias=True)
        nn.init.normal_(self.encoder.weight, std=0.01)

        # Initialize the decoder of the AutoRec model with values from the
        # normal distribution
        self.decoder = nn.Linear(num_hidden, num_users, bias=True)
        nn.init.normal_(self.decoder.weight, std=0.01)

        # Initialize the dropout module
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor: Tensor) -> Tensor:
        """The forward pass of the recommendation system.

        The sign of the forward pass of the recommendation system depends on
        whether or not the recommendation system is training or evaluating.

        Args:
            input_tensor: The input tensor to the forward pass

        Returns:
            The result of the forward pass of the recommendation system.
        """
        hidden: Tensor = self.dropout(self.encoder(input_tensor))
        pred: Tensor = self.decoder(hidden)

        # Result of forward pass depends on whether the model is training
        # or evaluating
        return pred * torch.sign(input_tensor) if self.training is True else pred
