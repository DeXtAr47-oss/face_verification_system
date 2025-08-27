"""
Face embedding network using Resnet50 model backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

class FaceEmbeddingNet(nn.Module):
    """
        CNN backbone for extracting face embeddings using ResNet50

        Args:
            embedding_dim (int): Dimension of the output embedding vector
            pretrained (bool): Choose wheather to use pretrained ResNet50 weights or not

    """

    def __init__(self, embedding_dim = 128):
        super(FaceEmbeddingNet, self).__init__()

        self.backbone = resnet50(weights = None)

        self.backbone.fc = nn.Identity()

        self.embedding = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p = 0.5),
            nn.Linear(512, embedding_dim)
        )
    
    def forward(self, x):
        """
            Forward pass through the network

            Args: 
                x (torch.tensor): input size of x (batch_size, 3, height, width)
            
            Return: 
                torch.tensor: L2-normalized embedding of shape (batch_size, embedding_dim)
        """
        x = self.backbone(x)
        embeddings = self.embedding(x)

        #applying l2 normalization on the embedding layer
        embeddings = F.normalize(embeddings, p = 2, dim = 1)
        return embeddings

