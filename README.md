# CIFAR-10 Image Classification with PyTorch

## Overview
This project implements a Convolutional Neural Network (CNN) using PyTorch to classify images from the CIFAR-10 dataset. The dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. The model is trained to classify images into one of the following categories:

- Airplane
- Automobile
- Bird
- Cat
- Deer
- Dog
- Frog
- Horse
- Ship
- Truck

## Dataset
The CIFAR-10 dataset is automatically downloaded and split into training and test sets.

## Implementation Steps
1. **Load and Normalize the CIFAR-10 Dataset**
   - Use `torchvision.datasets.CIFAR10` to load the dataset.
   - Apply transformations including normalization.
   - Use `DataLoader` to facilitate batch processing.

2. **Visualize Sample Data**
   - Display random training images using `matplotlib`.

3. **Define the CNN Model**
   - The model consists of:
     - Two convolutional layers (`Conv2d`)
     - Max pooling layers (`MaxPool2d`)
     - Fully connected layers (`Linear`)
     - Activation functions (`ReLU`)

4. **Define the Loss Function and Optimizer**
   - Use Cross-Entropy Loss (`nn.CrossEntropyLoss`) as the loss function.
   - Use Stochastic Gradient Descent (SGD) with momentum for optimization.

5. **Train the Model**
   - Train for multiple epochs.
   - Calculate loss at every 2000 mini-batches.

6. **Test the Model**
   - Predict labels for test images.
   - Compare predictions with ground truth.
   - Evaluate the overall model accuracy.

7. **Analyze Model Performance**
   - Compute accuracy for each class separately.

8. **Save the Trained Model**
   - Save model parameters using `torch.save()`.

9. **Train on GPU (if available)**
   - Move model and data to CUDA device for acceleration.

## Requirements
To run this project, install the following dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

## Future Improvements
- Increase model complexity (add more layers or use pretrained models like ResNet).
- Use data augmentation techniques to improve generalization.
- Fine-tune hyperparameters for better accuracy.
- Train for more epochs to improve performance.

---

This project provides a foundational approach to image classification using CNNs in PyTorch. 🚀

