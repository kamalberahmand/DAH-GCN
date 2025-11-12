import argparse
import torch
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from model import DAHGCN
from utils import evaluate


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset = Planetoid(root='./data', name=args.dataset.capitalize())
    data = dataset[0].to(device)
    
    model = DAHGCN(
        n_nodes=data.num_nodes,
        n_features=data.num_features,
        n_clusters=dataset.num_classes
    ).to(device)
    
    model.init_centroids(data.x, data.edge_index)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_nmi = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        
        z, q = model(data.x, data.edge_index)
        loss = model.loss(q)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                _, q = model(data.x, data.edge_index)
                pred = q.argmax(1).cpu().numpy()
                true = data.y.cpu().numpy()
                
                nmi, ari, acc = evaluate(true, pred)
                print(f'Epoch {epoch:3d} | Loss: {loss:.4f} | NMI: {nmi:.4f} | ARI: {ari:.4f} | ACC: {acc:.4f}')
                
                if nmi > best_nmi:
                    best_nmi = nmi
    
    print(f'\nBest NMI: {best_nmi:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    
    train(args)
