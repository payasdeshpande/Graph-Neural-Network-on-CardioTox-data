# Graph-Neural-Network-on-CardioTox-data
Implementation of 4 Graph Neural Network architectures on cardioTox data
the 4 architectures are :
1. Graph Convolutional Network (GCN)
2. Graph Attention Network(GAT)
3. GraphSAGE
4. Graph Isomorphic Network(GIN)

The Jupyter Notebook titled "GNN_Project" appears to be a comprehensive tutorial or project implementation involving Graph Neural Networks (GNNs) using the PyTorch Geometric library. Below is a detailed explanation for each significant part of the code which you can use for your GitHub repository documentation:

### Installation

The notebook begins by installing the required library:
```python
!pip install torch_geometric
```
This command installs `torch_geometric`, which is a library for deep learning on graph-structured data built upon PyTorch.

### Imports

The code imports various libraries necessary for handling graph data, creating graph neural networks, and processing:
```python
import tensorflow_datasets as tfds
import torch
from torch_geometric.data import Data
from torch_geometric.data import Batch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, GINConv, Sequential, global_mean_pool
from torch.nn import Linear, CrossEntropyLoss
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T
from torch.optim import Adam
import numpy as np
import matplotlib.pyplot as plt
```
These imports include TensorFlow Datasets for accessing the dataset, PyTorch for tensor computations and neural networks, and specific modules from PyTorch Geometric for building and training GNN models.

### Loading Data

There's a function to load a single graph from a dataset:
```python
def load_single_graph():
    ds, _ = tfds.load('cardiotox', split='train', with_info=True, as_supervised=False)
    for example in tfds.as_numpy(ds.take(1)):  # Taking just one sample for visualization
        node_features = torch.tensor(example['atoms'], dtype=torch.float)
        edge_indices_np = np.array(example['pair_mask'].nonzero())
        if edge_indices_np.shape[0] != 2:
            edge_indices_np = edge_indices_np.reshape(2, -1)
        edge_index = torch.tensor(edge_indices_np, dtype=torch.long)
        data = Data(x=node_features, edge_index=edge_index)
        return data
```
This function loads a graph from the "cardiotox" dataset and prepares it for use with PyTorch Geometric by creating a `Data` object which contains node features and edge indices.

### Graph Visualization

There's a visualization function to plot graphs:
```python
def plot_graph(data):
    G = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, node_size=700, node_color="skyblue", with_labels=True, font_weight='bold')
    plt.show()
```
This function converts the graph data into a NetworkX graph format and uses Matplotlib to visualize the graph structure.

### Training and Testing

The notebook provides a framework to train and evaluate graph neural networks:
```python
def train(model, optimizer, criterion, data_loader):
    # Function implementation
    # ...

def test(model, criterion, data_loader):
    # Function implementation
    # ...

def process_model(model_class):
    # Function implementation that includes training, testing, and logging of performance metrics
    # ...
```
These functions handle the training and testing phases for GNN models, including backpropagation and evaluation of accuracy and loss metrics.

### Model Execution

The notebook also includes commands to process specific GNN models:
```python
process_model(GAT)
process_model(GraphSAGE)
process_model(GIN)
```
Each call processes a different type of GNN model such as GAT, GraphSAGE, and GIN, using the previously defined `process_model` function.

This detailed breakdown can be used in your GitHub README or documentation to explain the structure, functionality, and purpose of the code in your "GNN_Project" notebook.
