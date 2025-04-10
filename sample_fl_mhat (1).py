import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import numpy as np
import flwr as fl
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define Multi-Layer Perceptron (MLP)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28*28, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)  # Corrected input size
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # Apply ReLU 
        x = torch.max_pool2d(torch.relu(self.conv2(x)), 2)  # Apply pooling
        x = x.view(x.size(0), -1)  # Flatten before FC layer
        x = torch.relu(self.fc1(x))  # Apply ReLU 
        return self.fc2(x)  # Output layer


# Define ResNet-18 (Fix: use weights=None instead of pretrained=False)
class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()
        self.model = models.resnet18(weights=None)  
        self.model.fc = nn.Linear(512, 10)  # Output layer for 10 classes

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # Convert 1-channel grayscale to 3-channel RGB
        return self.model(x)

# Define MobileNet (Fix: use weights=None)
class MobileNet(nn.Module):
    def __init__(self):
        super(MobileNet, self).__init__()
        self.model = models.mobilenet_v2(weights=None)  
        
        # Fix: Ensure first conv layer matches expected input
        self.model.features[0] = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        
        self.model.classifier[1] = nn.Linear(1280, 10)  

    def forward(self, x):
        x = x.repeat(1, 3, 1, 1)  # Convert grayscale (1 channel) to RGB (3 channels)
        return self.model(x)


# Load MNIST Dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
mnist_test = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

# Split into 4 subsets (one for each client)
num_clients = 4
num_samples = len(mnist_train) // num_clients
indices = list(range(len(mnist_train)))
random.shuffle(indices)

client_datasets = [Subset(mnist_train, indices[i * num_samples: (i + 1) * num_samples]) for i in range(num_clients)]
shared_dataset = Subset(mnist_train, indices[:2000])  # Shared public dataset

# Define Federated Learning Client
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=5):
        print(f"[CLIENT] Training started for {epochs} epochs on {device}...")
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()
            
            print(f"[CLIENT] Epoch {epoch+1}/{epochs} completed. Loss: {epoch_loss/len(self.train_loader):.4f}")
        
        print("[CLIENT] Training completed.")

    def get_parameters(self, config=None):
        print("[CLIENT] Sending model parameters to server...")
        return [param.cpu().detach().numpy() for param in self.model.parameters()]

    def set_parameters(self, parameters):
        print("[CLIENT] Receiving model parameters from server...")
        for param, new_param in zip(self.model.parameters(), parameters):
            param.data = torch.tensor(new_param).to(device)

    def evaluate(self, parameters, config):
        print("[CLIENT] Evaluating model on test dataset...")
        self.set_parameters(parameters)

        self.model.eval()
        correct = 0
        total = 0
        loss = 0.0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        loss = loss / len(self.test_loader)  # Average loss

        print(f"[CLIENT] Evaluation completed. Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")
        return loss, total, {"accuracy": accuracy}

    def fit(self, parameters, config):
        print("[CLIENT] Received training request from server. Starting training...")
        self.set_parameters(parameters)
        self.train(epochs=5)
        print("[CLIENT] Training completed. Sending updated model to server...")
        return self.get_parameters(), len(self.train_loader.dataset), {}


# Function to start each client separately
def start_client(client_id):
    models_dict = {0: MLP(), 1: CNN(), 2: ResNet18(), 3: MobileNet()}
    train_loader = DataLoader(client_datasets[client_id], batch_size=32, shuffle=True)
    test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)
    
    client = FLClient(models_dict[client_id], train_loader, test_loader)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

def start_server():
    print("[SERVER] Starting Flower Federated Learning server...")

    def get_evaluate_fn():
        """Returns a function for evaluating the global model on MNIST test set."""
        mnist_test = datasets.MNIST(root="./data", train=False, transform=transforms.ToTensor(), download=True)
        test_loader = DataLoader(mnist_test, batch_size=32, shuffle=False)

        def evaluate(server_round: int, parameters, config):
            print(f"[SERVER] Evaluating global model after ROUND {server_round}...")

            input_size = parameters[0].shape  # Check the shape of the first layer
            if len(input_size) == 2:  
                model = MLP()  # Fully connected MLP
            elif input_size[1] == 1:  
                model = CNN()  # CNN with 1-channel input
            elif input_size[1] == 3 and input_size[0] == 64:  
                model = ResNet18()  # ResNet expects 3-channel RGB input
            elif input_size[1] == 3 and input_size[0] == 32:  
                model = MobileNet()  # MobileNet expects 3-channel RGB input
            else:
                raise ValueError("[SERVER] Unknown model architecture!")

            model.to(device)
            model.eval()

            for param, new_param in zip(model.parameters(), parameters):
                param.data = torch.tensor(new_param).to(device)

            criterion = nn.CrossEntropyLoss()
            loss = 0.0
            correct, total = 0, 0

            with torch.no_grad():
                for batch in test_loader:  # Fix: Ensure correct unpacking
                    images, labels = batch  # Unpacking properly
                    images, labels = images.to(device), labels.to(device)
                    
                    if isinstance(model, MLP):
                        images = images.view(images.size(0), -1)  # Flatten for MLP

                    outputs = model(images)
                    loss += criterion(outputs, labels).item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = correct / total
            loss /= len(test_loader)

            print(f"[SERVER] Global model evaluation: Accuracy={accuracy:.4f}, Loss={loss:.4f}")
            return loss, {"accuracy": accuracy}

        return evaluate

    # Ensure at least 1 client is selected for training
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(),
        min_fit_clients=1,  
        min_available_clients=1,  
    )

    print("[SERVER] Waiting for clients to connect...")

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

    print("[SERVER] Training complete.")
