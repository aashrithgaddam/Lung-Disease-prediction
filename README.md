# Lung-Disease-prediction
Lung Disease Detection using CNNs An end-to-end deep learning pipeline to predict lung diseases (Pneumonia, COVID-19, Tuberculosis) from Chest X-Rays and CT scans using custom CNN architectures and transfer learning. Includes automated preprocessing, training scripts, and performance evaluation metrics.
# Lung Disease Prediction using Convolutional Neural Networks (CNN)

An end-to-end deep learning pipeline designed to automatically predict and classify lung diseases (such as Pneumonia, COVID-19, and Tuberculosis) from Chest X-Rays and CT scans. This repository utilizes state-of-the-art CNN architectures and Transfer Learning techniques to assist in automated medical image analysis.

---

## Key Features

* **Multi-Disease Classification:** Supports detection of Pneumonia, COVID-19, Tuberculosis, and Healthy lungs.
* **Hybrid Modeling:** Includes both a custom-built CNN architecture and pre-trained models via Transfer Learning (ResNet50, VGG16, EfficientNet).
* **Automated Preprocessing:** Standardized pipeline for image resizing, pixel normalization, and contrast enhancement (CLAHE).
* **Robust Data Augmentation:** Built-in strategies (rotation, zooming, horizontal flipping) to reduce overfitting on limited datasets.
* **Performance Analytics:** Automated generation of Confusion Matrices, ROC-AUC curves, and Precision-Recall metrics.

---

## Tech Stack & Libraries

* **Deep Learning Frameworks:** TensorFlow / Keras or PyTorch
* **Computer Vision:** OpenCV, Pillow
* **Data Science & Analytics:** NumPy, Pandas, Scikit-Learn
* **Visualization:** Matplotlib, Seaborn

---

##  Project Structure

```text
├── data/                   # Dataset directory (split into train/val/test)
├── notebooks/              # Jupyter Notebooks for experimentation & EDA
├── src/                    # Source code files
│   ├── preprocessing.py    # Image cleaning and augmentation pipeline
│   ├── models.py           # Custom CNN and transfer learning architectures
│   ├── train.py            # Training and evaluation orchestration
│   └── utils.py            # Helper modules (plotting, evaluation metrics)
├── models/                 # Saved weights and serialized models (.h5 or .pth)
├── requirements.txt        # Package dependencies
└── README.md               # Project documentation
```

---

##  Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/lung-disease-prediction-cnn.git
cd lung-disease-prediction-cnn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Preprocessing & Training
```bash
python src/train.py --epochs 25 --batch_size 32 --model resnet50
```

---

## Evaluation & Metrics

The models are evaluated using standard medical diagnostic metrics:
* **Accuracy:** Overall correctness of the model's classifications.
* **Sensitivity / Recall:** Critical for minimizing false negatives in medical diagnostics.
* **Precision:** Accuracy of positive disease predictions.
* **F1-Score:** Harmonic mean of precision and recall to balance evaluation.

---

## Disclaimer

This application is for educational and research purposes only. It is not intended to serve as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare professional regarding any clinical diagnosis.

