
import torch
import torch.nn as nn
from einops import einsum, rearrange

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim)
        self.vocab_size = num_embeddings
        self.d_model = embedding_dim
        self.w = nn.Parameter(nn.init.trunc_normal_(w, mean=0, std=1, a=-3, b=3))

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        # Look up the vector representation for each token ID in each sequence
        token_ids_expanded = rearrange(token_ids, "b s -> b s 1")
        # Create an array of [0, 1, 2, ..., vocab_size-1]
        class_values = torch.arange(self.vocab_size)
        class_values_expanded = rearrange(class_values, "v->1 1 v")
        one_hot_matrix = (token_ids_expanded == class_values_expanded).float()
        return einsum(one_hot_matrix, self.w, "b s v, v d_model -> b s d_model") 