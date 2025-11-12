import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.cluster import KMeans


class DAHGCN(nn.Module):
    def __init__(self, n_nodes, n_features, n_clusters, hidden=256, embed=128):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_clusters = n_clusters
        
        self.conv_s1 = GCNConv(n_features, hidden)
        self.conv_s2 = GCNConv(hidden, embed)
        self.S = nn.Parameter(torch.randn(n_nodes, embed) * 0.01)
        
        self.conv_t1 = GCNConv(n_features, hidden)
        self.conv_t2 = GCNConv(hidden, embed)
        
        self.fusion_weight = nn.Parameter(torch.tensor(0.5))
        self.centroids = nn.Parameter(torch.randn(n_clusters, embed))
    
    def forward(self, x, edge_index):
        z_s = F.relu(self.conv_s1(x, edge_index))
        z_s = self.conv_s2(z_s, edge_index)
        
        x_norm = F.normalize(x, p=2, dim=1)
        sim = torch.mm(x_norm, x_norm.t())
        topk_val, topk_idx = torch.topk(sim, k=10, dim=1)
        
        edge_t = []
        for i in range(self.n_nodes):
            for j in range(1, 10):
                edge_t.append([i, topk_idx[i,j].item()])
        edge_t = torch.tensor(edge_t).t().to(x.device)
        
        z_t = F.relu(self.conv_t1(x, edge_t))
        z_t = self.conv_t2(z_t, edge_t)
        
        beta = torch.sigmoid(self.fusion_weight)
        z = beta * z_s + (1 - beta) * z_t
        
        dist = torch.cdist(z, self.centroids)
        q = 1.0 / (1.0 + dist ** 2)
        q = q / q.sum(1, keepdim=True)
        
        return z, q
    
    def loss(self, q):
        p = q ** 2 / q.sum(0, keepdim=True)
        p = p / p.sum(1, keepdim=True)
        return F.kl_div(q.log(), p, reduction='batchmean')
    
    def init_centroids(self, x, edge_index):
        with torch.no_grad():
            z, _ = self.forward(x, edge_index)
            kmeans = KMeans(n_clusters=self.n_clusters, n_init=10)
            kmeans.fit(z.cpu().numpy())
            self.centroids.data = torch.tensor(kmeans.cluster_centers_).to(z.device)
