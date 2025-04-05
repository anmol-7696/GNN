import torch
from torch_geometric.datasets import Flickr
from torch_geometric.nn import GCNConv
import matplotlib.pyplot as plt
import networkx as nx

# Load the Flickr dataset, forcing a fresh download
dataset = Flickr(root='./data/Flickr', force_reload=True)
data = dataset[0]  # Single graph

# Define a simple GNN model
class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, 64)  # 500 features to 64
        self.conv2 = GCNConv(64, dataset.num_classes)  # 7 classes

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Initialize model, optimizer, and loss
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# Training loop
model.train()
for epoch in range(50):  # Short for demo
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# Evaluate and get predictions
model.eval()
with torch.no_grad():
    pred = model(data.x, data.edge_index).argmax(dim=1)

# Visualization (small subset)
G = nx.Graph()
subset_size = 50
for i in range(subset_size):
    G.add_node(i)
edges = data.edge_index.t().tolist()
for edge in edges:
    if edge[0] < subset_size and edge[1] < subset_size:
        G.add_edge(edge[0], edge[1])

# Color nodes by predicted labels
colors = [pred[i].item() for i in range(subset_size)]
pos = nx.spring_layout(G)
plt.figure(figsize=(10, 8))
nx.draw(G, pos, node_color=colors, with_labels=True, node_size=300, cmap=plt.cm.Set1)
plt.title("Flickr Graph (Subset) with GNN Predicted Categories")
plt.savefig('flickr_graph.png')  # Save for presentation
plt.show()