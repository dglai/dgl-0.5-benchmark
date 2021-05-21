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

from utils import Logger

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

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, edge_index):
        h = x
        for layer in self.layers:
            h = layer(h, edge_index)
        return h

def calc_acc(logits, labels, mask):
    logits = logits[mask]
    labels = labels[mask]
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, adj, labels, train_mask, val_mask, test_mask):
    model.eval()
    with torch.no_grad():
        logits = model(features, adj)
        train_acc = calc_acc(logits, labels, train_mask)
        val_acc = calc_acc(logits, labels, val_mask)
        test_acc = calc_acc(logits, labels, test_mask)
        return train_acc, val_acc, test_acc

def main(args):
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    path = osp.join('dataset', 'Reddit')
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

    logger = Logger(args.runs, args)
    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
                print('Training time/epoch {}'.format(np.mean(dur)))

            if not args.eval:
                continue

            train_acc, val_acc, test_acc = evaluate(model, features, adj, labels, train_mask, val_mask, test_mask)
            logger.add_result(run, (train_acc, val_acc, test_acc))

            print("Run {:02d} | Epoch {:05d} | Loss {:.4f} | Train {:.4f} | Val {:.4f} | Test {:.4f}".format(run, epoch, loss.item(), train_acc, val_acc, test_acc))

        if args.eval:
            logger.print_statistics(run)

    if args.eval:
        logger.print_statistics()


            
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
    parser.add_argument("--aggr", type=str, choices=['sum', 'mean'], default='mean',
                        help='Aggregation for messages')
    parser.add_argument("--weight-decay", type=float, default=5e-4,
                        help="Weight for L2 loss")
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    print(args)

    main(args)
