"""
    Siamese network for face verification
"""

import torch
import torch.nn as nn
from .embedding_net import FaceEmbeddingNet

class SiameseNet(nn.Module):
    """
        Args: 
            embedding_dim (int): Dimension of output embedding vector
    """

    def __init__(self, embedding_dim = 256):
        super(SiameseNet, self).__init__()
        self.embedding_net = FaceEmbeddingNet(embedding_dim=embedding_dim)

    def forward(self, x1, x2):
        """
            forward pass for a pair of images

            Args: 
                x1 (Torch.Tensor): First image tensor
                x2 (Torch.Tensor): Second image tensor
            
            Return: 
                Tuple [Torch.Tensor, Torch.Tensor]: embedding for both the images
        """
        embedding1 = self.embedding_net(x1)
        embedding2 = self.embedding_net(x2)

        return embedding1, embedding2   

    def get_embedding(self, x):
        """
            Embedding for a single image

            Args: 
                x Torch.Tensor: Image tensor
            
            Return: 
                Torch.Tensor: embedding vector
        """
        x = self.embedding_net(x)
        return x

    