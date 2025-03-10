import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class QM9_GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(QM9_GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 32)  
        self.conv2 = GCNConv(32, 32)  
        self.conv3 = GCNConv(32, num_classes)  

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)
