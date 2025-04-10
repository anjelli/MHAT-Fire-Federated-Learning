# Federated Learning under Model and Data Heterogeneity
# Federated Learning Experiments with Flower

This repository demonstrates how federated learning can be applied using the Flower framework to two different use cases:

1. **Model Heterogeneity with MNIST Dataset** (`sample_fl_mhat.py`)
2. **Wildfire Detection with Satellite Data** (`sample_fl_fire.py`)

## 📁 Contents

- `sample_fl_mhat.py`: Trains MLP, CNN, ResNet18, and MobileNet independently across four FL clients on the MNIST dataset.
- `sample_fl_fire.py`: Uses Random Forest (Turkey), SVM (India), and MLP (USA) to perform wildfire classification on MODIS 2010 data.

## 🔧 Requirements

Install required libraries:
```bash
pip install flwr torch torchvision pandas scikit-learn
