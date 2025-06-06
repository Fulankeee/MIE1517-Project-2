# MIE1517-Project-2 Hand Gesture Recognition with CNNs

The goal of this project is to design and train a custom **Convolutional Neural Network (CNN)** that can classify **American Sign Language (ASL) gestures** (letters A–I) based on image input.

---

## Project Files

| File                          | Description                                                                 |
|-------------------------------|-----------------------------------------------------------------------------|
| `MIE1517_A2.ipynb`            | Full notebook with code, visualization and results              |
| `Hand Gesture.zip` | Zip file of 27 cleaned images (3 per class for A–I), collected manually     |
| `A2_Hand_Gesture_Unlabled_Data.zip` | Zip file of unlabeled images |
| `README.md`                   | This file, explaining project setup and key takeaways                      |

---
Run this notebook directly in **Google Colab** (recommended).
> [Open in Colab](https://colab.research.google.com/github/Fulankeee/MIE1517-Project-2/blob/main/A2.ipynb#scrollTo=X6WDvajSqIDs)

## Part A – Data Collection

- **Objective**: Collect a small dataset of hand gestures from scratch.
- **Classes**: 9 letters – A, B, C, D, E, F, G, H, I.
- **Image Count**: 3 images per letter → **27 images total**.
- **Tools used**: Mobile phone + manual cropping tools.
- **Preprocessing**:
  - Resized all images to **224×224 pixels**
  - Ensured hand was centered and background was clean
  - Converted to RGB `.jpg` format with consistent naming (`<student_id>_A_1.jpg`)

---

## Part B – CNN-Based Gesture Classifier
### Dataset & Preprocessing
- **Transforms Applied**:
  ```python
  transforms.Compose([
      transforms.Resize((224, 224)),
      transforms.ToTensor()
  ]) 
- **DataLoader Batch Size: 32**

### CNN Architecture
Custom CNN built using `torch.nn`:

```python
Conv2d(3 to 32) to ReLU to MaxPool
Conv2d(32 →to 64) to ReLU to MaxPool
Flatten to Linear(64*56*56 to 32) to ReLU to Linear(32 to 9)
```
- Loss: `CrossEntropyLoss`
- Optimizer: `Adam`
- Epochs: 5
- Accuracy Achieved: **84.1% on test set**

### Transfer Learning with AlexNet
- Used pretrained `AlexNet` for feature extraction
- Achieved **~92.3%** test accuracy

### Transfer Learning with VGG16
- Used pretrained `VGG16` (deep 16-layer CNN)
- Achieved highest test accuracy: **95.5%**
- Most accurate on gesture I (100%)
- Most confusion: A vs E, F vs G

## Model Comparison

| Model                | Test Accuracy (%) |
|----------------------|-------------------|
| Custom CNN (scratch) | 84.1%             |
| AlexNet              | 92.3%            |
| VGG16                | **95.5%**         |

## Discussion
- Transfer learning improves accuracy with limited training data.
- Data augmentation, batch norm, or deeper training would improve base CNN.
- VGG16 provided robust performance but is computationally heavier.
- Bias may exist due to lighting, hand skin tone, or gesture similarity.