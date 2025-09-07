""" Training scripts for face verification model. """

import argparse
import os
from src import face_verification

def main():
    parser = argparse.ArgumentParser(description="Train Face verification model")
    parser.add_argument("--train-dir", type=str, required=True,
                        help="Directory containing training data")
    
    parser.add_argument("--val-dir", type=str, default=None,
                        help="Directory containing validation data")
    
    parser.add_argument("--output-dir", type=str, default="models",
                        help="Directory to save the trainned model")

    parser.add_argument("--epochs", type=int, default=1,
                        help="No. of training steps")
    
    parser.add_argument("--batch-sixe", type=int, default=16,
                        help="Batch size for training")
    
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    
    parser.add_argument("--embedding-dim", type=int, default=256,
                        help="Embedding dimension")
    
    parser.add_argument("--device", type=str, default="auto",
                        help="Change the device to (cuda) or (cpu) according to usecase")
    
    args = parser.parse_args()

    # make output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # initilize system
    device = "mps" if args.device == "auto" else args.device

    system = face_verification(
        embedding_dim = args.embedding_dim,
        device = device
    )

    print("Training face verification model....")
    print(f"Training directory {args.train_dir}")
    print(f"Validation directory {args.val_dir}")
    print(f"Device = {system.device}")
    
    train_losses = system.train(
        train_dir = args.train_dir,
        val_dir = args.val_dir,
        epochs = args.epochs,
        batch_sizr = args.batch_size,
        lr = args.lr
    )

    # save trainned model
    model_path = os.path.join(args.output_dir, "face_verification.pth")
    system.save_model(model_path)

    print(f"training completed")
    print(f"model saved to {model_path}")

if __name__ == "__main__":
    main()
