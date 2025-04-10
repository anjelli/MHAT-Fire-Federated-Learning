import torch
import torch.nn as nn
import torch.optim as optim
import flwr as fl
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset
import joblib
import io

# Load and preprocess data
def load_data(file_path):
    print(f"[DATA] Loading dataset: {file_path}")
    df = pd.read_csv(file_path)
    df.rename(columns={"type": "class"}, inplace=True)

    categorical_columns = ["satellite", "instrument", "daynight"]
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    df.drop(columns=["acq_date", "acq_time", "version"], inplace=True, errors="ignore")
    X = df.drop(columns=["class"])
    y = df["class"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Load datasets
X_train_turkey, X_test_turkey, y_train_turkey, y_test_turkey = load_data("data/modis_2010_Turkey.csv")
X_train_india, X_test_india, y_train_india, y_test_india = load_data("data/modis_2010_India.csv")
X_train_usa, X_test_usa, y_train_usa, y_test_usa = load_data("data/modis_2010_USA.csv")

# Convert USA data to PyTorch tensors
X_train_usa = torch.tensor(X_train_usa, dtype=torch.float32)
y_train_usa = torch.tensor(y_train_usa.values, dtype=torch.long)
X_test_usa = torch.tensor(X_test_usa, dtype=torch.float32)
y_test_usa = torch.tensor(y_test_usa.values, dtype=torch.long)

usa_train_loader = DataLoader(TensorDataset(X_train_usa, y_train_usa), batch_size=32, shuffle=True)
usa_test_loader = DataLoader(TensorDataset(X_test_usa, y_test_usa), batch_size=32, shuffle=False)

# Define MLP model for the USA
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

mlp_model = MLP(X_train_usa.shape[1], len(set(y_train_usa.numpy())))
mlp_optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
mlp_criterion = nn.CrossEntropyLoss()

def train_mlp():
    mlp_model.train()
    for X_batch, y_batch in usa_train_loader:
        mlp_optimizer.zero_grad()
        outputs = mlp_model(X_batch)
        loss = mlp_criterion(outputs, y_batch)
        loss.backward()
        mlp_optimizer.step()

def evaluate_mlp():
    mlp_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in usa_test_loader:
            outputs = mlp_model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
    return correct / total

# Train RFC for Turkey
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train_turkey, y_train_turkey)

def evaluate_rfc():
    return rfc.score(X_test_turkey, y_test_turkey)

# Train SVM for India
svm = SVC(probability=True)
svm.fit(X_train_india, y_train_india)

def evaluate_svm():
    return svm.score(X_test_india, y_test_india)

# Define FLClient
class FLClient(fl.client.NumPyClient):
    def __init__(self, model, eval_fn, model_type):
        self.model = model
        self.eval_fn = eval_fn
        self.model_type = model_type
    
    def get_parameters(self, config=None):
        return [] if self.model_type != "MLP" else [param.detach().cpu().numpy() for param in self.model.parameters()]
    
    def set_parameters(self, parameters):
        if self.model_type == "MLP":
            for param, new_param in zip(self.model.parameters(), parameters):
                param.data = torch.tensor(new_param, dtype=param.dtype)
    
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        acc = self.eval_fn()
        print(f"[CLIENT] Evaluation accuracy: {acc:.4f}")
        return 0.0, 100, {"accuracy": acc}

# Start clients
def start_client(client_id):
    if client_id == 0:
        client = FLClient(rfc, evaluate_rfc, "RFC")  # Turkey
    elif client_id == 1:
        client = FLClient(svm, evaluate_svm, "SVM")  # India
    elif client_id == 2:
        client = FLClient(mlp_model, evaluate_mlp, "MLP")  # USA
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

# Start server
def start_server():
    print("[SERVER] Starting FL server...")

    def get_evaluate_fn():
        def evaluate(server_round, parameters, config):
            print(f"[SERVER] Evaluating model after ROUND {server_round}...")
            acc_usa = evaluate_mlp()
            acc_turkey = evaluate_rfc()
            acc_india = evaluate_svm()
            avg_acc = (acc_usa + acc_turkey + acc_india) / 3
            print(f"[SERVER] Federated Evaluation Accuracy: {avg_acc:.4f}")
            return 0.0, {"accuracy": avg_acc}
        return evaluate
    
    strategy = fl.server.strategy.FedAvg(
        evaluate_fn=get_evaluate_fn(),
        min_fit_clients=1,
        min_available_clients=1,
    )
    fl.server.start_server(server_address="localhost:8080", config=fl.server.ServerConfig(num_rounds=5), strategy=strategy)
