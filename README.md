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

- **Model Architecture**:
 ```python
 class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(64 * 56 * 56, 32)
        self.fc2 = nn.Linear(32, 9)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
- Conv Layers: Extract spatial features
- ReLU: Introduces non-linearity
- MaxPool: Downsamples to reduce feature map size
- FC1 to FC2: Classifies into 9 output classes



