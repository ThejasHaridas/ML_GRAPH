import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load the MATLAB data
def load_data(file_path):
    mat_data = scipy.io.loadmat(file_path)
    # We have 62 channels of EEG data
    features = []
    for i in range(1, 63):
        channel_data = mat_data[f'EEg_Fthk_{i}']
        features.append(channel_data)
    
    # Stack all channels together
    features = np.stack(features, axis=2)  # Shape: (1716, 5046, 62)
    
    # Create labels (11 classes, 156 samples each)
    labels = np.repeat(np.arange(11), 156)
    
    return features, labels

# Create graph structure from EEG data
def create_graph_data(eeg_data, labels):
    graph_list = []
    num_channels = eeg_data.shape[2]  # 62 channels
    
    # Calculate correlation between channels
    print("Calculating channel correlations...")
    # Reshape data to compute correlation between channels
    # Shape: (num_samples * time_points, num_channels)
    flat_data = eeg_data.reshape(-1, num_channels)
    correlation_matrix = np.corrcoef(flat_data.T)
    
    # Create edge indices based on correlation threshold
    edge_index = []
    correlation_threshold = 0.75  # Increased threshold to create stronger connections
    
    # Create edges between highly correlated channels
    for i in range(num_channels):
        for j in range(i+1, num_channels):
            if abs(correlation_matrix[i, j]) >= correlation_threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])  # Add both directions for undirected graph
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t()
    print(f"Created {len(edge_index[0])//2} unique edges based on correlation threshold {correlation_threshold}")
    
    # Normalize the data
    mean = np.mean(eeg_data, axis=(0, 1), keepdims=True)
    std = np.std(eeg_data, axis=(0, 1), keepdims=True)
    eeg_data = (eeg_data - mean) / (std + 1e-8)
    
    # Create graph objects for each sample
    print("Creating graph objects...")
    for i in range(len(eeg_data)):
        # Use the time series data as node features
        x = torch.tensor(eeg_data[i].T, dtype=torch.float)
        
        # Add edge weights based on correlation values
        edge_weights = []
        for edge in edge_index.t():
            weight = abs(correlation_matrix[edge[0], edge[1]])
            edge_weights.append(weight)
        edge_weights = torch.tensor(edge_weights, dtype=torch.float)
        
        data = Data(x=x,
                   edge_index=edge_index,
                   edge_attr=edge_weights,  # Add edge weights
                   y=torch.tensor([labels[i]], dtype=torch.long))
        graph_list.append(data)
    
    return graph_list

# Define the GNN model
class EEG_GNN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(EEG_GNN, self).__init__()
        # Reduce feature dimensionality first
        self.feature_conv = nn.Conv1d(1, 32, kernel_size=100, stride=20)
        reduced_size = (num_node_features - 100) // 20 + 1
        
        # GCN layers with edge weights
        hidden_channels = 128
        self.conv1 = GCNConv(32 * reduced_size, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, 64)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.bn3 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(64, num_classes)
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Feature reduction
        x = x.unsqueeze(1)
        x = F.relu(self.feature_conv(x))
        x = x.view(x.size(0), -1)
        
        # GCN layers with edge weights
        x = self.conv1(x, edge_index, edge_weight=edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x, edge_index, edge_weight=edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x, edge_index, edge_weight=edge_attr)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Global pooling
        x = global_mean_pool(x, batch)
        x = self.dropout(x)
        x = self.fc(x)
        
        return F.log_softmax(x, dim=1)

def train_model(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test_model(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            output = model(data)
            pred = output.max(1)[1]
            correct += pred.eq(data.y.view(-1)).sum().item()
            total += data.y.size(0)
    return correct / total

def main():
    # Load data
    print("Loading data...")
    features, labels = load_data('EEG_Fthk_62.mat')
    
    print("Creating graph data...")
    graph_data = create_graph_data(features, labels)
    
    # Split data
    train_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)  
    test_loader = DataLoader(test_data, batch_size=16)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = EEG_GNN(num_node_features=5046, num_classes=11).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    
    # Training loop
    print("Starting training...")
    epochs = 200  
    train_losses = []
    test_accuracies = []
    
    best_acc = 0.0
    patience = 20  
    no_improve = 0
    
    for epoch in range(epochs):
        train_loss = train_model(model, train_loader, optimizer, device)
        test_acc = test_model(model, test_loader, device)
        
        train_losses.append(train_loss)
        test_accuracies.append(test_acc)
        
        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
        else:
            no_improve += 1
            
        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Test Accuracy: {test_acc:.4f}')
        
        scheduler.step(test_acc)
    
    print(f"\nBest Test Accuracy: {best_acc:.4f}")
    
    # Plot training results
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies)
    plt.title('Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.close()

if __name__ == "__main__":
    main()
