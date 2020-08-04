import argparse
import dgl
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.data import load_data
from dgl.nn.pytorch import GATConv

class GAT(nn.Module):
    def __init__(self,
                 g,
                 num_layers,
                 in_feats,
                 num_hidden,
                 num_classes,
                 heads,
                 activation=F.elu,
                 feat_drop=0.0,
                 attn_drop=0.0,
                 negative_slope=0.2):
        super(GAT, self).__init__()

        self.num_layers = num_layers
        self.g = g
        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(GATConv(in_feats=in_feats,
                                       out_feats=num_hidden,
                                       num_heads=heads[0],
                                       feat_drop=0.,
                                       attn_drop=0.,
                                       negative_slope=negative_slope,
                                       activation=activation))
        # hidden layers
        for l in range(num_layers - 2):
            # due to multi-head, the in_feats = num_hidden * num_heads
            self.gat_layers.append(GATConv(in_feats=num_hidden * heads[l],
                                           out_feats=num_hidden,
                                           num_heads=heads[l + 1],
                                           feat_drop=feat_drop,
                                           attn_drop=attn_drop,
                                           negative_slope=negative_slope,
                                           activation=activation))
        # output projection
        self.gat_layers.append(GATConv(in_feats=num_hidden * heads[-2],
                                       out_feats=num_classes,
                                       num_heads=heads[-1],
                                       feat_drop=feat_drop,
                                       attn_drop=attn_drop,
                                       negative_slope=negative_slope,
                                       activation=None))

    def reset_parameters(self):
        for layer in self.gat_layers:
            layer.reset_parameters()

    def forward(self, h):
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](self.g, h).flatten(1)
        logits = self.gat_layers[-1](self.g, h).mean(1)
        return logits

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data = load_data(args)
    features = torch.FloatTensor(data.features)
    labels = torch.LongTensor(data.labels)
    if hasattr(torch, 'BoolTensor'):
        train_mask = torch.BoolTensor(data.train_mask)
        val_mask = torch.BoolTensor(data.val_mask)
        test_mask = torch.BoolTensor(data.test_mask)
    else:
        train_mask = torch.ByteTensor(data.train_mask)
        val_mask = torch.ByteTensor(data.val_mask)
        test_mask = torch.ByteTensor(data.test_mask)
    in_feats = features.shape[1]
    n_classes = data.num_labels
    n_edges = data.graph.number_of_edges()
    print("""----Data statistics------'
      #Edges %d
      #Classes %d
      #Train samples %d
      #Val samples %d
      #Test samples %d""" %
          (n_edges, n_classes,
           train_mask.int().sum().item(),
           val_mask.int().sum().item(),
           test_mask.int().sum().item()))

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    # Remove duplicate edges
    # In PyG, this is a default pre-processing step for Reddit, see
    # https://github.com/rusty1s/pytorch_geometric/blob/master/torch_geometric/datasets/reddit.py#L58
    g = data.graph
    g = dgl.add_self_loop(g)
    g = g.int().to(device)
    features, labels = features.to(device), labels.to(device)

    model = GAT(g=g,
                num_layers=args.num_layers,
                in_feats=in_feats,
                num_hidden=args.num_hidden,
                num_classes=n_classes,
                heads=[1, 1, 1],
                feat_drop=args.dropout,
                attn_drop=args.dropout)
    model = model.to(device)

    loss_fcn = nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # initialize graph
    dur = []
    for epoch in range(args.epochs):
        model.train()
        if epoch >= 3:
            t0 = time.time()
        # forward
        logits = model(features)
        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if args.eval:
            acc = evaluate(model, features, labels, val_mask)
        else:
            acc = 0
        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                            acc, n_edges / np.mean(dur) / 1000))

    if args.eval:
        print()
        acc = evaluate(model, features, labels, test_mask)
        print("Test Accuracy {:.4f}".format(acc))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--dataset", type=str, default='reddit')
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-layers", type=int, default=3,
                        help="number of hidden layers")
    parser.add_argument("--lr", type=float, default=0.0029739421726400865,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=2.4222556964495987e-05,
                        help="weight decay")
    parser.add_argument("--num-hidden", type=int, default=16,
                        help="number of hidden units")
    parser.add_argument("--dropout", type=float, default=0.18074706609292976,
                        help="Dropout to use")
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    main(args)
