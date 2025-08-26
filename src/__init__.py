"""
    Face verification system a production level code
"""

__version__ = "1.0.0"

from .models import FaceEmbeddingNet, SiameseNet
from .data import TripletDataset, FaceDataset
from .face_verification import FaceVerification

__all__ = ["FaceEmbeddingNet", "SiameseNet", "TripletDataset", "FaceDataset", "FaceVerification"]