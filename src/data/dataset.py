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
        transform(callable): Transform functionality to apply to images.
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.pairs = []
        self.allimgs = []
        self.person_to_images = {}  # Track images by person for better negative sampling

        # function to load the dataset
        self._load_dataset(root_dir)
        
        # Print dataset info
        print(f"Loaded {len(self.pairs)} anchor-positive pairs")
        print(f"Total images available: {len(self.allimgs)}")

    def _load_dataset(self, root_dir):
        print(f"Loading dataset from: {root_dir}")
        
        if not os.path.exists(root_dir):
            raise ValueError(f"Root directory does not exist: {root_dir}")
        
        # Check if root directory has any contents
        contents = os.listdir(root_dir)
        print(f"Root directory contents: {contents}")
        
        if not contents:
            raise ValueError(f"Root directory is empty: {root_dir}")
        
        # traverse each of the person name in the folder
        valid_persons = 0
        for person_folder in os.listdir(root_dir):
            person_path = os.path.join(root_dir, person_folder)

            # Skip hidden files and non-directories
            if person_folder.startswith('.') or not os.path.isdir(person_path):
                print(f"Skipping: {person_folder} (not a directory)")
                continue
            
            print(f"Processing person: {person_folder}")
            
            # List contents of person folder
            person_contents = os.listdir(person_path)
            print(f"  Person folder contents: {person_contents}")
            
            # store the path of anchor images
            anchor_imgs = []
            for f in os.listdir(person_path):
                # Skip directories and hidden files when looking for anchor images
                if f.startswith('.') or f == 'distortion':
                    continue
                    
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    file_path = os.path.join(person_path, f)
                    if os.path.isfile(file_path):
                        anchor_imgs.append(file_path)
                        print(f"    Found anchor image: {f}")

            # store the path of positive images
            distorted_folder = os.path.join(person_path, 'distortion')
            print(f"  Looking for distortion folder at: {distorted_folder}")
            
            if not os.path.isdir(distorted_folder):
                print(f"  Warning: No distortion folder found for {person_folder}")
                # Try alternative names
                for alt_name in ['distorted', 'augmented', 'transformed']:
                    alt_path = os.path.join(person_path, alt_name)
                    if os.path.isdir(alt_path):
                        print(f"  Found alternative folder: {alt_name}")
                        distorted_folder = alt_path
                        break
                else:
                    continue

            distorted_imgs = []
            if os.path.isdir(distorted_folder):
                distortion_contents = os.listdir(distorted_folder)
                print(f"    Distortion folder contents: {distortion_contents}")
                
                for f in os.listdir(distorted_folder):
                    if f.lower().endswith(('.jpg', '.png', '.jpeg')):
                        file_path = os.path.join(distorted_folder, f)
                        if os.path.isfile(file_path):
                            distorted_imgs.append(file_path)
                            print(f"      Found distorted image: {f}")

            if not anchor_imgs:
                print(f"  Warning: No anchor images found for {person_folder}")
                continue
                
            if not distorted_imgs:
                print(f"  Warning: No distorted images found for {person_folder}")
                continue

            # Store images by person for negative sampling
            self.person_to_images[person_folder] = anchor_imgs + distorted_imgs

            # make pair of anchor and positive images and store it in a list
            pairs_count = 0
            for anc_img in anchor_imgs:
                for pos_img in distorted_imgs:
                    self.pairs.append((anc_img, pos_img, person_folder))
                    pairs_count += 1

            # store all the distorted images 
            for img in distorted_imgs:
                if img not in self.allimgs:
                    self.allimgs.append(img)

            # store the anchor images
            for img in anchor_imgs:
                if img not in self.allimgs:
                    self.allimgs.append(img)
            
            print(f"  Successfully processed {person_folder}: {len(anchor_imgs)} anchor, {len(distorted_imgs)} distorted, {pairs_count} pairs")
            valid_persons += 1
        
        print(f"Dataset loading complete:")
        print(f"  Valid persons: {valid_persons}")
        print(f"  Total pairs: {len(self.pairs)}")
        print(f"  Total images: {len(self.allimgs)}")
        
        if len(self.pairs) == 0:
            raise ValueError("No valid anchor-positive pairs found! Check your data structure:\n"
                           "Expected structure:\n"
                           "root_dir/\n"
                           "  person1/\n"
                           "    anchor1.jpg\n"
                           "    anchor2.jpg\n"
                           "    distortion/\n"
                           "      distorted1.jpg\n"
                           "      distorted2.jpg\n")

    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        anchor_path, pos_path, current_person = self.pairs[idx]

        # Load anchor and positive images
        try:
            anchor_img = Image.open(anchor_path).convert('RGB')
            positive_img = Image.open(pos_path).convert('RGB')
        except Exception as e:
            print(f"Error loading anchor/positive images: {e}")
            raise

        # Select negative image from a different person
        # Get all other persons
        other_persons = [p for p in self.person_to_images.keys() if p != current_person]
        
        if not other_persons:
            # Fallback: select from all images except current pair
            available_negatives = [img for img in self.allimgs 
                                 if img != anchor_path and img != pos_path]
        else:
            # Select from images of other persons
            neg_person = random.choice(other_persons)
            available_negatives = self.person_to_images[neg_person]
        
        if not available_negatives:
            # Last resort fallback
            available_negatives = [img for img in self.allimgs 
                                 if img != anchor_path and img != pos_path]
        
        if not available_negatives:
            raise ValueError("No negative images available!")
        
        neg_path = random.choice(available_negatives)
        
        try:
            neg_img = Image.open(neg_path).convert('RGB')
        except Exception as e:
            print(f"Error loading negative image {neg_path}: {e}")
            raise

        if self.transform:
            anchor_img = self.transform(anchor_img)
            positive_img = self.transform(positive_img)
            neg_img = self.transform(neg_img)

        return anchor_img, positive_img, neg_img