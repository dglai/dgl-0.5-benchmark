import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator
from torch.nn import Parameter
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.nn.inits import glorot, zeros

from utils import Logger

class SAGEConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(SAGEConv, self).__init__()

        self.in_feats = in_feats
        self.out_feats = out_feats

        self.weight = Parameter(torch.Tensor(in_feats, out_feats))
        self.root_weight = Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = Parameter(torch.Tensor(out_feats))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        glorot(self.root_weight)
        zeros(self.bias)

    def forward(self, x, adj):
        out = adj.matmul(x, reduce="mean") @ self.weight
        out = out + x @ self.root_weight + self.bias
        return out


class GraphSAGE(nn.Module):
    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 num_layers,
                 dropout):
        super(GraphSAGE, self).__init__()

        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(SAGEConv(in_feats, hidden_feats))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(SAGEConv(hidden_feats, hidden_feats))
        # output layer
        self.layers.append(SAGEConv(hidden_feats, out_feats))
        self.dropout = nn.Dropout(p=dropout)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x, adj):
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, adj)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, adj)

        return x.log_softmax(dim=-1)

def train(model, x, adj, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(x, adj)[train_idx]
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, x, adj, y_true, split_idx, evaluator):
    model.eval()

    out = model(x, adj)
    y_pred = out.argmax(dim=-1, keepdim=True)

    train_acc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['acc']
    valid_acc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['acc']
    test_acc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['acc']

    return train_acc, valid_acc, test_acc

def main():
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GraphSAGE Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-products')
    split_idx = dataset.get_idx_split()

    data = dataset[0]
    edge_index = data.edge_index.to(device)
    edge_index = to_undirected(edge_index, data.num_nodes)
    adj = SparseTensor(row=edge_index[0], col=edge_index[1])

    x, y_true = data.x.to(device), data.y.to(device)
    train_idx = split_idx['train'].to(device)

    model = GraphSAGE(in_feats=data.x.size(-1),
                      hidden_feats=args.hidden_channels,
                      out_feats=dataset.num_classes,
                      num_layers=args.num_layers,
                      dropout=args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-products')
    logger = Logger(args.runs, args)

    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(model, x, adj, y_true, train_idx, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))
            if not args.eval:
                continue

            result = test(model, x, adj, y_true, split_idx, evaluator)
            logger.add_result(run, result)

            if epoch % args.log_steps == 0:
                train_acc, valid_acc, test_acc = result
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_acc:.2f}%, '
                      f'Valid: {100 * valid_acc:.2f}% '
                      f'Test: {100 * test_acc:.2f}%')

        if args.eval:
            logger.print_statistics(run)
    if args.eval:
        logger.print_statistics()


if __name__ == "__main__":
    main()
