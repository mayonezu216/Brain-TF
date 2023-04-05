import torch
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from dataset.HCP import HCPDataset,HCP_ave_Dataset
# Load the Cora dataset
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
from math import ceil
from torch_geometric.nn import DenseGraphConv,  GCNConv
from DMoNPool import DMoNPooling
from torch_geometric.utils import to_dense_adj, to_dense_batch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = HCPDataset(root='/fast/beidi/Com-BrainTF/dataset/rest1_AP')
dataset_ave = HCP_ave_Dataset(root='/fast/beidi/Com-BrainTF/dataset/rest1_AP')
# print(dir(dataset))

train_num = int(1*len(dataset))
val_num = int(0.3*len(dataset))
avg_num_nodes = 24
train_data = dataset[:train_num]
val_data =  dataset[train_num:]
test_data =  dataset_ave
train_loader = DataLoader(train_data, batch_size=1, shuffle=True)

val_loader = DataLoader(val_data, batch_size=1, shuffle=False)

# Define the GCN model

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(268, hidden_channels,cached=True,normalize=True)
        self.conv2 = GCNConv(hidden_channels, hidden_channels,cached=True,normalize=True)
        # self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 2)

    def forward(self, x, edge_index, batch, edge_weight):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index,edge_weight)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index,edge_weight)

        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
    
        x = self.lin(x)
        return x

class Net(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super().__init__()

        self.conv1 = GCNConv(in_channels, hidden_channels,cached=True,normalize=True)
        num_nodes = ceil(0.5 * avg_num_nodes)
        # print(num_nodes)
        self.pool1 = DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv2 = DenseGraphConv(hidden_channels, hidden_channels)
        num_nodes = ceil(0.5 * num_nodes)
        self.pool2 = DMoNPooling([hidden_channels, hidden_channels], num_nodes)

        self.conv3 = DenseGraphConv(hidden_channels, hidden_channels)

        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch,edge_weight):
        # print(x.size())
        # print(edge_index.size())
        # print(edge_weight.size())
       
        x = self.conv1(x, edge_index,edge_weight).relu()

        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, batch)

        s, x, adj, sp1, o1, c1,clu = self.pool1(x, adj, mask)
        # print(clu)
        x = self.conv2(x, adj).relu()

        s, x, adj, sp2, o2, c2,clu1 = self.pool2(x, adj)
        
        # Cluster nodes based on learned assignments
        # print(x,x.size()) #ba, , fea_dim
        
        # print(_,_.size())
        # print(adj.size())
        # cluster_labels = s.argmax(dim=-2).squeeze()
        # print(cluster_labels)
        # assert 2==3

        x = self.conv3(x, adj)

        x = x.mean(dim=1)
        x = self.lin1(x).relu()
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1), sp1 + sp2 + o1 + o2 + c1 + c2, clu

# Initialize the model and optimizer

# model = GCN(hidden_channels=64)
model = Net(268,2)
criteria = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train the model

for epoch in range(10):
    model.train()
    # cluster_labels = []
    for data in train_loader: 
        optimizer.zero_grad()
        # out = model(data.x, data.edge_index, data.batch) 
        # print(data.edge_index.size())
        
        out, tot_loss,cluster_label = model(data.x, data.edge_index, data.batch,data.edge_attr)
        # print(cluster_label)
        # data.x, data.edge_index, data.batch)
        loss = criteria(out, data.y) 
        loss.backward()
        optimizer.step()
    # cluster_labels = torch.tensor(cluster_labels)
    # cluster_labels = torch.squeeze(cluster_labels)

    print('Epoch: {:03d}, Loss: {:.4f}'.format(epoch, loss.item()))

    # # Evaluate the model on the test set
    # model.eval()
    # for data in val_loader: 
    #     # out = model(data.x, data.edge_index, data.batch)
        
    #     pred, tot_loss,cluster_label = model(data.x, data.edge_index, data.batch,data.edge_attr)
    #     # print(cluster_label)
    #     loss = F.nll_loss(pred, data.y) 


    #     pred = pred.argmax(dim=1)
        
    #     correct = pred.eq(data.y).sum().item()
    #     acc = correct / data.y.size(0)
        # print('Test Accuracy: {:.4f}'.format(acc))
count = 0
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
for data in test_loader: 
    count += 1
    print(count)
    # data.x = data.x.view(688,268,-1)
    # data.x = torch.mean(data.x,dim=0)
    
    # data.edge_index = data.edge_index.view(2,688,-1)
    # data.edge_index = torch.mean(data.edge_index,dim=2)
    
    # data.edge_attr = data.edge_attr.view(688,-1)
    # data.edge_attr = torch.mean(data.edge_attr,dim=2)
    print(data.x.size())
    print(data.edge_attr.size())
    print(data.edge_index.size())
    print(data.batch.size())
    pred, tot_loss,cluster_label = model(data.x, data.edge_index, data.batch,data.edge_attr)
    print(cluster_label)
    assert 2==3