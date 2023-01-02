"""Create the matrix factorization model."""
import torch
import torch.nn as nn
from torch import Tensor


class MatrixFactorization(nn.Module):
    """The matrix factorization model for the prupose of recommendations."""

    user_embeddings: nn.Embedding
    item_embeddings: nn.Embedding
    user_bias: nn.Embedding
    item_bias: nn.Embedding

    def __init__(
        self, num_factors: int, num_users: int, num_items: int, **kwargs
    ) -> None:
        """Initialize the various layers of the Matrix Factoriztaion model.
        
        Args:
            num_factors: The number of factors used in the model.
            num_users: The number of users considered in the model.
            num_items: The number of items considered in the model.
        """
        super(MatrixFactorization, self).__init__(**kwargs)

        # Initialize user embeddings and bias with values drawn from normal 
        # distribution
        self.user_embeddings = nn.Embedding(num_users, num_factors)
        nn.init.normal_(self.user_embeddings.weight, std=0.01)

        self.user_bias = nn.Embedding(num_users, 1)
        nn.init.normal_(self.user_bias.weight, std=0.01)

        # Initialize user embeddings with values drawn from normal distribution
        self.item_embeddings = nn.Embedding(num_items, num_factors)
        nn.init.normal_(self.item_embeddings.weight, std=0.01)

        self.item_bias = nn.Embedding(num_items, 1)      
        nn.init.normal_(self.item_bias.weight, std=0.01)
        
    def forward(self, user_id: int, item_id: int) -> Tensor:
        """Execute the forward pass of the matrix factorization model.
        
        Args:
            user_id: The numerical ID of the user.
            item_id: The numerical ID of the item.
        
        Returns:
            The output tensor of the forward pass.
        """
        # Run the user ID through the user embeddings and bias
        P_u: Tensor = self.user_embeddings(user_id)
        b_u: Tensor = self.user_bias(user_id)

        # Run the item ID through the item embeddings bias
        Q_i: Tensor = self.item_embeddings(item_id)
        b_i: Tensor = self.item_bias(item_id)

        # Multiply the output of the embeddings together and add the biases
        output: Tensor = torch.sum(P_u * Q_i, axis=1) + torch.squeeze(b_u) + torch.squeeze(b_i)
        
        return output
