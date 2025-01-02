import torch
from torch import nn

class CrossAttentionLatentSpaceEmbedding(nn.Module):
    """
    """

    def __init__(self, D, d):
        """
        Keywords
        --------
        D: int
            Number of spikes
        d: int
            Number of dimensions in the latent space
        """

        super().__init__()

        self.Wq = nn.Linear(D, D)
        self.Wk = nn.Linear(d, D)
        self.Wv = nn.Linear(d, D)

        return
    
    def forward(self, X):
        """
        """

        Z0 = torch.tensor(X, dtype=torch.float32)
        

        return