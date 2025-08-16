"""
    Image transformation for face verification
"""

import torchvision.transforms as transforms

def train_transformation(image_size = 224):
    """
    Get training transforms with data augmentation

    Args: 
        image_size(int): Target size image
    
    Returns: 
        transforms.Compose: Training transforms
    
    """

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def val_transformation(image_size = 224):
    """
    Get validation transforms without augmentation

    Args: 
        image_size(int): Target image size
    
    Returns: 
        transforms.Compose: validation transforms

    """
    
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])