from src.face_verification import FaceVerification 
import torch
import os

def main():
    system = FaceVerification()

    train_dir = "/Users/pritamdas/python/dataset/Task_B/train"
    val_dir = "/Users/pritamdas/python/dataset/Task_B/val"

    losses = system.train(train_dir=train_dir, val_dir=val_dir)

    system.save_model(file_path="/Users/pritamdas/python/projects/face_verification_system/model")

if __name__ == "__main__":
    main()
    
