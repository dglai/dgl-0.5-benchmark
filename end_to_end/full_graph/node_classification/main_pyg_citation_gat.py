import os.path as osp

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch_geometric.nn import MessagePassing, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch_geometric.transforms as T

from utils import Logger

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_feats,
                 num_hidden,
                 num_classes,
                 heads,
                 activation=F.elu,
                 dropout=0.):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_feats,
                                       num_hidden,
                                       heads=heads[0],
                                       dropout=0.))

        # hidden layers
        for l in range(num_layers - 2):
            # due to multi-head, the in_feats = num_hidden * num_heads
            self.gat_layers.append(GATConv(num_hidden * heads[l],
                                           num_hidden,
                                           heads=heads[l + 1],
                                           dropout=dropout))
        # output projection
        self.gat_layers.append(GATConv(num_hidden * heads[-2],
                                       num_classes,
                                       heads=heads[-1],
                                       concat=False,
                                       dropout=dropout))
        self.activation = activation

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        for l in range(self.num_layers):
            x = self.gat_layers[l](x, adj)
            if l != self.num_layers - 1:
                x = self.activation(x)
        return x

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

def main():
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help="weight decay")
    parser.add_argument("--num-hidden", type=int, default=8,
                        help="number of hidden units")
    parser.add_argument("--dropout", type=float, default=.6,
                        help="Dropout to use")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    parser.add_argument("--runs", type=int, default=10)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    path = osp.join('dataset', args.dataset)
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
    data = dataset[0]

    features = data.x.to(device)
    labels = data.y.to(device)
    edge_index = data.edge_index.to(device)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])
    train_mask = torch.BoolTensor(data.train_mask).to(device)
    val_mask = torch.BoolTensor(data.val_mask).to(device)
    test_mask = torch.BoolTensor(data.test_mask).to(device)

    model = GAT(num_layers=args.num_layers,
                in_feats=features.size(-1),
                num_hidden=args.num_hidden,
                num_classes=dataset.num_classes,
                heads=[8, 8, 1],
                dropout=args.dropout).to(device)


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
    main()
