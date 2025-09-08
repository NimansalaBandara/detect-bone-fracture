# Bone Fracture Detection Using CNN

This project uses a **Convolutional Neural Network (CNN)** to detect bone fractures from X-ray images. The model classifies images into two categories: **fractured** and **non-fractured** bones.

---

## Dataset

The dataset consists of X-ray images organized into **training, validation, and test sets**. Each set contains two folders: `fractured` and `non-fractured`. Images were preprocessed to **grayscale** and resized to **128x128 pixels** for efficient training.

---

## Data Preprocessing

- Images are read in grayscale.
- Resized to 128x128 pixels.
- Invalid or unreadable images are removed.
- Processed images and labels are stored in structured arrays for training and testing.

---

## Model Architecture

The CNN consists of:

1. **Conv2D layers**: 64 and 32 filters with 3x3 kernel, ReLU activation.
2. **MaxPooling2D layers**: 2x2 pool size.
3. **Flatten layer**: Converts feature maps into a 1D vector.
4. **Dense layers**: 128 and 64 neurons with ReLU activation.
5. **Output layer**: 1 neuron with sigmoid activation for binary classification.

---

## Training

- Optimizer: **Adam** (learning rate 0.001)
- Loss: **Binary Crossentropy**
- Epochs: 10
- Batch size: 16
- Callbacks used:
  - **EarlyStopping** (patience 10)
  - **ReduceLROnPlateau**
  - **ModelCheckpoint** (save best model)

The model achieved high accuracy on both training and validation sets, showing good generalization.

---

## Results

- **Training Accuracy**: ~99%
- **Validation Accuracy**: ~98%
- **Test Accuracy**: 98.84%
- **Test Loss**: 0.054

The model demonstrates strong performance in detecting bone fractures from X-ray images.

---

## Inference

You can predict a single image using:

```python
predict_class('path_to_image.png')
