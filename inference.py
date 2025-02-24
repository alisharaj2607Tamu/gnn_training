import torch
from model import GCN
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

dataset = Planetoid(root='data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

# Load trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(num_node_features=dataset.num_node_features, num_classes=dataset.num_classes).to(device)
model.load_state_dict(torch.load('models/best_model.pth'))
model.eval()

# Perform inference
with torch.no_grad():
    logits = model(data.to(device))
    predictions = logits.argmax(dim=1)

print("Predicted labels:", predictions.tolist())
