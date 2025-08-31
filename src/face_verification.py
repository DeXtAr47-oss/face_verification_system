import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
from PIL import Image
import os


from .models import SiameseNet
from .data import TripletDataset, FaceDataset, train_transformation, val_transformation


class FaceVerification:
    """
        FaceVerification class to train the model over the siamese network
    
    Args:
        embedding_dim (int): accepted embedding dimension over the siamese network
        device (str): use device (mps) or (cpu)
        margin (float): margin for triplet loss 
        
    """
    def __init__(self, embedding_dim = 256, device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu'), margin = 0.5):
        self.embedding_dim = embedding_dim
        self.device = device

        # initilize the model
        self.model = SiameseNet(embedding_dim=embedding_dim).to(self.device)

        #use pytorch's built in TripletMarginLoss 
        self.triplet_loss = nn.TripletMarginLoss(margin=margin, p = 2, eps = 1e-6, swap = True, reduction = "mean")

        # use transformation for training
        self.train_transform = train_transformation()

        # use transformation for validation
        self.val_transform = val_transformation()

    def train(self, train_dir, val_dir, epochs = 5, batch_size = 16, lr = 0.001):
        """
            Main training funtion to train the siamese network model

        Args: 
            train_dir (str): directory containing training data
            val_dir (str): directory containing validation data
            epochs (int): number of training epochs
            batch_size (int): batch size for training
            lr (float): learning rate

        Return: 
            list:  training loss per epoch
        """

        #create datasets
        train_dataset = TripletDataset(root_dir=train_dir, transform=self.train_transform)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

        #optimizer and scheduler
        optimizer = torch.optim.Adam(self.model.parameters(), lr = lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.5)

        #training loop
        self.model.train()
        train_loss = []

        for e in range(epochs):
            epoch_loss = 0.0
            num_batches = 0

            progress_bar = tqdm(train_dataloader, desc = f"Epoch {e + 1}/{epochs}")

            for idx, (anchor, positive, negative) in enumerate(progress_bar):

                #move the images to device
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                #set gradients to zero
                optimizer.zero_grad()

                #get the embeddings for anchor, positive and negative images
                anchor_embd = self.model.get_embedding(anchor)
                positive_embd = self.model.get_embedding(positive)
                negative_embd = self.model.get_embedding(negative)

                # compute tripleloss and update the gradients
                loss = self.triplet_loss(anchor_embd, positive_embd, negative_embd)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                progress_bar.set_postfix({"loss": f"{loss.item(): .4f}"})
            
            avg_loss = epoch_loss / num_batches
            train_loss.append(avg_loss)
            scheduler.step()

            print(f"Epoch {e + 1}/{epochs} completed, loss: {avg_loss: .4f}")

            if val_dir:
                val_acc = self.validate(val_dir)
                print(f'validation accuracy: {val_acc: .4f}')

        return train_loss
    
    def validate(self, val_dir, threshold = 0.6):
        """
            Validate the model on verification pairs

            Args:
                val_dir (str): Directory containing validation data.
                threshold (int): Similarity threshold for verificaiton.

            Returns:
                float: validation accuracy
        """
        self.model.eval()

        # create validation dataset
        val_dataset = FaceDataset(root_dir = val_dir, transform = self.val_transform)

        # generate positive and negative pairs
        positive_pairs = []
        negative_pairs = []

        # group images by identity
        identity_to_images = {}

        for i, (_, label, path) in enumerate(val_dataset):
            if label not in identity_to_images:
                identity_to_images[label] = []
            identity_to_images[label].append((i, path))

        #generate positive pairs
        for identity, images in identity_to_images.items():
            if len(images) >= 2:
                for i in range(len(images)):
                    for j in range(i + 1, len(images)):
                        positive_pairs.append((images[i][0], images[j][0], 1))

        # generate negative pairs 
        identities = list(identity_to_images.keys())
        for i in range(min(len(positive_pairs), 1000)):
            id1, id2 = random.sample(identities, 2)
            img1 = random.choice(identity_to_images[id1])[0]
            img2 = random.choice(identity_to_images[id2])[0]
            negative_pairs.append((img1, img2, 0))

        all_pairs = positive_pairs + negative_pairs
        predictions = []
        labels = []

        with torch.no_grad():
            for idx1, idx2, label in tqdm(all_pairs, desc="Validating"):
                img1, _, _ = val_dataset[idx1]
                img2, _, _ = val_dataset[idx2]

                img1 = img1.unsqueeze(0).to(self.device)
                img2 = img2.unsqueeze(0).to(self.device)

                emb1 = self.model.get_embedding(img1)
                emb2 = self.model.get_embedding(img2)

                distance = F.pairwise_distance(emb1, emb2).item()
                prediction = 1 if distance < threshold else 0

                predictions.append(prediction)
                labels.append(label)

        accuracy = accuracy_score(labels, predictions)
        return accuracy * 100
    
    def verify_identity(self, test_image_path, identity_folder_path, thershold = 0.6):
        """
            Verify if test image belongs to the given identity folder

            Args: 
                test_image_path (str): path to the test image
                identity_folder_path (str): path to the identity folder
                threshold (float): similarity thershold
            
            Returns:
                tuple: (is_match, confidence_score, closest_distance)
        """
        
        self.model.eval()

        # load test image
        test_img = Image.open(test_image_path).convert('RGB')
        test_tensor = self.val_transform(test_img).unsqueeze(0).to(self.device)

        # get test image embedding
        with torch.no_grad():
            test_embedding = self.model.get_embedding(test_tensor)

        # load all images from the identity folder
        identity_imgs = [f for f in os.listdir(identity_folder_path)
                         if f.lower().endswith(('.png','.jpg','.jpeg'))]
        
        min_distance = float('inf')
        distances = []

        with torch.no_grad():
            for img_name in identity_imgs:
                img_path = os.path.join(identity_folder_path, img_name)
                ref_img = Image.open(img_path).convert('RGB')
                ref_tensor = self.val_transform(ref_img).unsqueeze(0).to(self.device)

                ref_embedding = self.model.get_embedding(ref_tensor)
                distance = F.pairwise_distance(test_embedding, ref_embedding).item()
                distances.append(distance)
                min_distance = min(min_distance, distance)

        is_match = min_distance < thershold
        confidence = 1.0 - min_distance

        return (is_match, confidence, min_distance)
    
    def save_model(self, file_path):
        """save the trainned model"""

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'embedding_dim': self.embedding_dim
        }, file_path)

    def load_model(self, file_path):
        """load a trainned model"""

        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("model loaded from: {file_path}")
