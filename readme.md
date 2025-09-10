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
The `TripletDataset` class and `FaceDataset` class expect the follwing folder structure to train on the distotion images.
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
       └── (same structure as train)
```
## 🛠️ Command line useage
## Training
```bash
    python scripts/train.py \
        --train-dir data/train \
        --val-dir  data/validation \
        --epochs 5 \
        --batch-size 16 \
        --lr 0.001 \
        --embedding-dim 256 \
        --output-dir models/
```

## Single Image Verification
```bash
    python scripts/inference.py \
        --model-path models/face_verification_model.pth \
        --test-image test_image.jpg \
        --identity-folder identity_folder \
        --threshold 0.6
```

## Batch Evaluation
```bash
    python scripts/evaluate.py \
        --model-path models/face-verification.pth \
        --test-dir data/test
        --identity-dir data/identities
```

## 🏛️ Architecture
## Model Components
* __Embedding_net__ `(src/models/embedding_net.py)`
    * ResNet50 backbone.
    * Custom embedding layers: 2048 -> 512 -> 256.
    * L2 normalization for unit sphere embeddings.

* __SiameseNet__ `(src/models/siamese_ent.py)`
    * Shared weights for consistent embeddings.
    * Dual input processing for pairs.
    * Single embedding extraction method.

* __Faceverification__ `(src/face_verification.py)`
    * Main system orchestrating training and inference.
    * Uses PyTorch's `nn.TripletMrginLoss`.
    * Comprehensive validation and evaluation methods.

## Data Pipeline
1. __TripletDataset__`(src/data/datasets.py)`
    * Automatically generates (anchor, positive, negative) triplets.
    * Handles identity folders with minimum requirements.
    * Dynamic triplet during training.

2. __FaceDataset__ `(src/data/datasets.py)`
    * Standard dataset for validation and training.
    * Identity-based organization.
    * Robust image loading with error handling.

3. __Transforms__ `(src/data/transforms.py)`
    * Training: Data augmentation (flip, rotation, color jitter).
    * Validation: Standard preprocessing only.
    * ImagNet normalizaion.

## ⚙️ Key classes and methods
## Faceverification
```python
    class face_verification:
        def __init__(self, embedding_dim=256, device="auto", margin=0.5)
        def train(self, train_dir, val_dir, epoch=5, lr=0.001)
        def validate(self, val_dir, threshold=0.6)
        def save_model(self, filepath)
        def load_model(self, filepath)
        def evaluate_test_set(self, test_dir, identity_folder_dir, threshold=0.6)
```
## Key features
* __Triplet Loss__ `(nn.TripletMarginLoss(margin=0.5, p=2, reduction='mean'))`
* __Progressive training__: Learning rate scheduling with StepLR.
* __Validation__: Automatic positive/negative pair generation for accuracy measurement.
* __Batch Processing__: Efficient batch inference with progress tracking.
* __Model Persistence__: Save/Load functionallity with metadata.

## 📈 Example Useage
__Example 1__: Basic training and inference
```python
    from src import face_verification

    # initilize and train
    system = face_verification(embedding_dim=256)
    losses = system.train('data/train', 'data/val', epochs=10)
    system.save_model('path/to/model.pth')

    # load and verify
    system = face_verification()
    system.load_model('path/to/model.pth')
    result = system.verify_identity('test.jpg', 'person1_folder/', threshold=0.6)
    print(f"Result: {result}")
```

__Example 2__: Batch evaluation
```python

    # evaluate entire test set
    results = system.evaluate_test_set(
        test_dir = "test/image",
        identity_folder_dir = "identity/",
        threshold = 0.6
    )

    # calculate accuracy
    total = len(results)
    matched = sum(1 for r in results if r['is_match'])
    accuracy = matched/total
    print(f"accuracy: {accuracy: .3f}")
```

## Configuration Options
## Model Parameters
* `embedding_dim`: 64, 128, 256, 512 (default 256)
* `margin`: 0.1-1.0 for triplet loss (default: 0.5)
* `device`: 'cuda', 'mps', 'cpu', 'auto' (default: auto='mps')

## Training Parameters
* `epochs`: Number of training epochs (default: 5)
*  `batch_size`: Training batch size (default: 16)
* `lr`: Learning rate (default: 0.001)

## Inference parameters
* `threshold`: Similarity threshold for matching (default: 0.6)
    * Lower = stricher match.
    * Higher = more lenient matching.

## Requirments
```bash
    torch>=1.9.0
    torchvision>=0.10.0
    numpy>=1.21.0
    opencv-python>=4.5.0
    Pillow>=8.3.0
    scikit-learn>=1.0.0
    matplotlib>=3.4.0
    tqdm>=4.62.0
```
## 🤝 Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create Pull requests.

## If this project helps you please give it a star ⭐


 

    