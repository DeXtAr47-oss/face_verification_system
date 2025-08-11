"""
    Nwural network models for face verification
"""

from .embedding_net import FaceEmbeddingNet
from .siamese_net import SiameseNet

__all__ = ["FaceEmbeddingNet", "SiameseNet"]