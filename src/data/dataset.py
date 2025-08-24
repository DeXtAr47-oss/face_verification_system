"""
    Dataset classes for face verification 
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset

class FaceDataset(Dataset):
    """
        Dataset class for loading face images organized in validation folder.

        Args: 
            root_dir (str): Root directory containing validation images.
            transform (callable): transform function to apply to the images.
    """

    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir
        self.transform = transform

        # get all the identity folders
        identities = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

        # build image paths and labels
        self.img_paths = []
        self.labels = []

        for idx, identity in enumerate(identities):
            anchor_path = os.path.join(root_dir, identity)
            anchor_imgs = [f for f in os.listdir(anchor_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

            positive_path = os.path.join(anchor_path, 'distortion')
            pos_imgs = [f for f in os.listdir(positive_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
            for img in anchor_imgs:
                self.img_paths.append(os.path.join(anchor_path, img))

                for img in pos_imgs:
                    self.img_paths.append(os.path.join(positive_path, img))
            
                self.labels.append(idx)

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        
        return image, label, img_path

class TripletDataset(Dataset):
    """
        Dataset that generates triplets (anchor, positive, negative) images.

        Args:
            root_dir(str): Root directory containing the image folders
            transform(callable): Transform functionalliry to apply to images.
    """

    def __init__(self, root_dir, transform = None):
        self.transform = transform
        self.pairs = []
        self.allimgs = []

        # function to load the dataset
        self._load_dataset(root_dir)

    def _load_dataset(self, root_dir):

        # traverse each of the person naem in the folder
        for person_folder in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_folder)

            # if person path is not found the return
            if not os.path.isdir(person_path):
                continue
            
            # store the path of anchor images
            anchor_imags = []
            for f in os.listdir(person_path):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(person_path, f)

                    # check for valid file
                    if os.path.isfile(file_path):
                        anchor_imags.append(file_path)

            # store the path of positive images
            distorted_folder = os.path.join(person_path, 'distortion')
            if not os.path.isdir(distorted_folder):
                continue

            distorted_imags = []
            for f in os.listdir(distorted_folder):
                if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                    file_path = os.path.join(distorted_folder, f)
                    if os.path.isfile(file_path):
                        distorted_imags.append(file_path)

            # make pair of anchor and positive images and store it in a list
            for anc_img in anchor_imags:
                for pos_img in distorted_imags:
                    self.pairs.append((anc_img, pos_img))

            # store all the other positve images 
            for img in distorted_folder:
                if img not in self.allimgs:
                    self.allimgs.append(img)

            # store the anchor images
            for img in anchor_imags:
                if img not in self.allimgs:
                    self.allimgs.append(img)

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_path, pos_path = self.pairs[idx]

        anchor_img = Image.open(anchor_path).convert('RGB')
        positive_img = Image.open(pos_path).convert('RGB')

        neg_path = random.choice([img for img in self.allimgs if img != anchor_path and img != pos_path])
        neg_img = Image.open(neg_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            neg_img = self.transform(neg_img)

        return anchor_img, positive_img, neg_img


     
        