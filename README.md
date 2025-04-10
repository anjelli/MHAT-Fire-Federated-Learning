# Federated Learning under Model and Data Heterogeneity
# Federated Learning Experiments with Flower

This repository demonstrates how federated learning can be applied using the Flower framework to two different use cases:

1. **Model Heterogeneity with MNIST Dataset** (`sample_fl_mhat.py`)
2. **Wildfire Detection with Satellite Data** (`sample_fl_fire.py`)

## 📁 Contents

- `sample_fl_mhat.py`: Trains MLP, CNN, ResNet18, and MobileNet independently across four FL clients on the MNIST dataset.
- `sample_fl_fire.py`: Uses Random Forest (Turkey), SVM (India), and MLP (USA) to perform wildfire classification on MODIS 2010 data.

This project investigates how federated learning performs when both the models and data vary across clients—an increasingly common scenario in real-world applications. Using the Flower framework, we simulate federated training across multiple clients with different machine learning models and diverse datasets.

Two experimental setups are explored:

Image Classification (MNIST): Clients independently train MLP, CNN, ResNet18, and MobileNet models on non-overlapping subsets of the MNIST dataset.

Wildfire Detection (MODIS 2010): Clients from three regions (Turkey, India, USA) train models on local satellite data using Random Forest, SVM, and MLP respectively.

These experiments demonstrate the flexibility of federated learning in handling heterogeneity—whether it's in the form of model architecture or data type. The goal is to understand how collaborative learning can succeed even when clients are anything but uniform.

