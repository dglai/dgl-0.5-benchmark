import os.path as osp

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch_geometric.datasets import Reddit
from torch_geometric.nn.inits import glorot, zeros

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggr,
                 feat_drop=0.,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_feats = in_feats
        self._out_feats = out_feats
        self._aggr = aggr
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation

        self.weight = Parameter(torch.Tensor(in_feats, out_feats))
        self.root_weight = Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = Parameter(torch.Tensor(out_feats))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root_weight)
        zeros(self.bias)

    def forward(self, x, adj):
        x = self.feat_drop(x)
        if self._aggr == 'sum':
            out = adj.matmul(x) @ self.weight
        elif self._aggr == 'mean':
            out = adj.matmul(x, reduce="mean") @ self.weight
        else:
            return ValueError("Expect aggregation to be 'sum' or 'mean', got {}".format(self._aggr))
        out = out + x @ self.root_weight + self.bias
        if self.activation is not None:
            out = self.activation(out)
        return out

class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 aggr,
                 activation=F.relu,
                 dropout=0.):
        super(GraphSAGE, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(SAGEConv(in_feats, n_hidden, aggr, activation=activation))
        self.layers.append(SAGEConv(n_hidden, n_classes, aggr, feat_drop=dropout, activation=None))

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

def evaluate(model, adj, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    path = osp.join('..', 'data', 'Reddit')
    dataset = Reddit(path)
    data = dataset[0]

    features = data.x.to(device)
    labels = data.y.to(device)
    edge_index = data.edge_index.to(device)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    val_mask = torch.BoolTensor(data.val_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)

    model = GraphSAGE(dataset.num_features,
                      args.n_hidden,
                      dataset.num_classes,
                      args.aggr,
                      F.relu,
                      args.dropout).to(device)

    loss_fcn = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    dur = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features, adj)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if args.eval:
            acc = evaluate(model, adj, features, labels, val_mask)
        else:
            acc = 0
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} ".format(
            epoch, np.mean(dur), loss.item(), acc))

    if args.eval:
        print()
        acc = evaluate(model, adj, features, labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GraphSAGE')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--dropout", type=float, default=0.5,
                        help="dropout probability")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--epochs", type=int, default=200,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=16,
                        help="number of hidden gcn units")
    parser.add_argument("--aggr", type=str, choices=['sum', 'mean'], default='sum',
                        help='Aggregation for messages')
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    main(args)
