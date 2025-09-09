# Face verification system 
A production ready face verification system using deep learning metric and Siamese network for robust identity verification. Built with PyTorch and designed for real world applications. This project is made for verifying distorted images with the anchor images for a given dataset, for more reference about the siamese network refer to the [paper](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf).

# Features
* __Metric Learning__ - Uses pytorch optimized TripleMarginLoss for embedding space learning.
* __Generilization__ -  Works with identitites never seen while trainning (no fixed classses).
* __Robust__ - Handles image distortion with data augnemtation.
* __Fast__ - ResNet50 backbone with efficient embedding layers.
* __Production ready__ - Modular code structure comprehensive testing.
* __Comprehensive__ - Training validation and batch evaluation tools.

## 🚀 Quick Start
## Installation
```bash
    # clone this repo
    git clone https://github.com/DeXtAr47-oss/face_verification_system

    # install all the required packages
    pip install -r requirements.txt
```

## Basic Useage
```python

    from src import face_verification

    # inintilize system
    system = face-verification(
        embedding_dim = 256,
        device = "cuda",
        margin = 0.5
    )

    # train the resnet50 model on personal data
    train_losses = system.train(
        train_dir = "path/to/train_dir",
        val_dir = "path/to/val_dir",
        epochs = 5,
        batch_size = 16,
        lr = 0.001
    )

    # save the model 
    system.save_model(file_path = "path/to/save_dir/model.pth")

    # load inference
    system.load_model(file_path = "path/to/load_dir/model.pth")

    # verify identity
    is_match, confidence, min_distance = system.verify_identity(test_image_path = "/path/to/test_img", 
                                                                identity_folder_path = "/path/to/identity_folder",
                                                                threshold = 0.6)
    
    print(f"Match: {is_match}, Confidence: {confidence: .3f}, Distance: {min_distance: .3f}")
```

## Data Organization
```text
    data/
    ├── train/
    │ ├── person_001/
    │ │ ├── anchor.jpg
    │ │ └── distortion/
    │ │ ├── distortion_001.jpg
    │ │ ├── distortion_002.jpg
    │ │ └── ...
    │ ├── person_002/
    │ │ ├── anchor.jpg
    │ │ └── distortion/
    │ │ ├── distortion_001.jpg
    │ │ └── ...
    │ └── ...
    ├── validation/
    │ └── (same structure as train)
```


    