import torch
import torch.nn.functional as F
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from torch.optim import Adam
from torchmetrics.regression import MeanAbsoluteError

# Loading QM9 dataset
dataset = QM9(root='data')

# Determining number of node features and target properties
num_features = dataset.num_node_features  
num_targets = dataset[0].y.shape[0]  

# Train-Validation-Test split
torch.manual_seed(42)
num_train = int(0.8 * len(dataset))
num_val = int(0.1 * len(dataset))
num_test = len(dataset) - num_train - num_val

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_val, num_test])

# Data Loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Defining GCN Model
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        # Aggregate node features correctly for batch processing
        x = global_mean_pool(x, data.batch)  
        return x  




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_features, hidden_dim=64, output_dim=1).to(device)

# Optimizer & Loss
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
mae_metric = MeanAbsoluteError().to(device)

# Training function
def train():
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data).squeeze()  
        loss = F.mse_loss(out, data.y[:, 0])  
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# Evaluation function (using MAE)
def evaluate(loader):
    model.eval()
    total_loss = 0
    total_mae = 0
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data).squeeze()
            loss = F.mse_loss(out, data.y[:, 0])
            total_loss += loss.item()
            
            mae = mae_metric(out, data.y[:, 0])
            total_mae += mae.item()

    avg_loss = total_loss / len(loader)
    avg_mae = total_mae / len(loader)
    return avg_loss, avg_mae

# Training loop
best_val_loss = float('inf')
for epoch in range(50):
    train_loss = train()
    val_loss, val_mae = evaluate(val_loader)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'models/qm9_best_model.pth')  # Save best model

    if epoch % 5 == 0:
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val MAE: {val_mae:.4f}")

# Final evaluation
test_loss, test_mae = evaluate(test_loader)
print(f"Final Test Loss: {test_loss:.4f} | Final Test MAE: {test_mae:.4f}")
