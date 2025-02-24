import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import NormalizeFeatures
from torch.optim import Adam
from model import GCN

# Load the Cora dataset
dataset = Planetoid(root='data', name='Cora', transform=NormalizeFeatures())

# Defining model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)
data = dataset[0].to(device)

# Optimizer
optimizer = Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# Training function
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])  # Loss only on training nodes
    loss.backward()
    optimizer.step()
    return loss.item()

# Evaluation function
def evaluate():
    model.eval()
    with torch.no_grad():
        logits = model(data)
        pred = logits.argmax(dim=1)  # Get class with highest probability
        acc = (pred[data.test_mask] == data.y[data.test_mask]).sum().item() / data.test_mask.sum().item()
    return acc

# Training loop
best_acc = 0
for epoch in range(200):
    loss = train()
    acc = evaluate()
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), 'models/best_model.pth')  # Save best model

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Loss: {loss:.4f} | Test Accuracy: {acc:.4f}")

print(f"Best Accuracy: {best_acc:.4f}")
