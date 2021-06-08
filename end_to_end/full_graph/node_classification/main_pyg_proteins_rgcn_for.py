import argparse
import math
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor

from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils import Logger

class RelGraphConv(nn.Module):
    r"""Relational graph convolution layer.

    This is a variant of RelGraphConv for ogbn-proteins:
    1. We incorporate prior edge weights for weighting messages
    2. ogbn-proteins is a special relational graph where we have edges
       of all relations between each pair of connected nodes.

    As a result, it looks very similar to GATConv.
    """
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_relations,
                 activation=None,
                 dropout=0.):
        super(RelGraphConv, self).__init__()

        self._rel_fcs = nn.ParameterList([
            nn.Parameter(torch.Tensor(in_feats, out_feats)) for _ in range(num_relations)
        ])
        self._skip = nn.Linear(in_feats, out_feats, bias=True)
        self._activation = activation
        self._dropout = nn.Dropout(dropout)

    def reset_parameters(self):
        for rel_fc in self._rel_fcs:
            nn.init.kaiming_uniform_(rel_fc, a=math.sqrt(5))
        self._skip.reset_parameters()

    def forward(self, node_feats, adjs):
        relation_node_feats = []
        for rel, rel_adj in enumerate(adjs):
            relation_node_feats.append(rel_adj.matmul(node_feats, reduce="mean") @ self._rel_fcs[rel])
        new_node_feats = torch.stack(relation_node_feats, dim=0).sum(0)
        node_feats = new_node_feats + self._skip(node_feats)
        if self._activation:
            node_feats = self._activation(node_feats)
        node_feats = self._dropout(node_feats)

        return node_feats

class RGCN(nn.Module):
    def __init__(self,
                 num_layers,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 num_relations,
                 activation=F.relu,
                 dropout=0.):
        super(RGCN, self).__init__()

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(RelGraphConv(in_feats=in_feats,
                                            out_feats=hidden_feats,
                                            num_relations=num_relations,
                                            activation=activation,
                                            dropout=dropout))
        for l in range(num_layers - 2):
            self.gnn_layers.append(RelGraphConv(in_feats=hidden_feats,
                                                out_feats=hidden_feats,
                                                num_relations=num_relations,
                                                activation=activation,
                                                dropout=dropout))
        self.gnn_layers.append(RelGraphConv(in_feats=hidden_feats,
                                            out_feats=out_feats,
                                            num_relations=num_relations,
                                            dropout=0.))

    def reset_parameters(self):
        for gnn in self.gnn_layers:
            gnn.reset_parameters()

    def forward(self, node_feats, adjs):
        for gnn in self.gnn_layers:
            node_feats = gnn(node_feats, adjs)
        return node_feats

def train(model, node_feats, adjs, y_true, train_idx, optimizer):
    model.train()
    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer.zero_grad()
    out = model(node_feats, adjs)[train_idx]
    loss = criterion(out, y_true[train_idx].to(torch.float))
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, node_feats, adjs, y_true, split_idx, evaluator):
    model.eval()

    y_pred = model(node_feats, adjs)

    train_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['train']],
        'y_pred': y_pred[split_idx['train']],
    })['rocauc']
    valid_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['valid']],
        'y_pred': y_pred[split_idx['valid']],
    })['rocauc']
    test_rocauc = evaluator.eval({
        'y_true': y_true[split_idx['test']],
        'y_pred': y_pred[split_idx['test']],
    })['rocauc']

    return train_rocauc, valid_rocauc, test_rocauc

def main():
    parser = argparse.ArgumentParser(description='OGBN-Proteins (RGCN Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--num-layers', type=int, default=3)
    parser.add_argument('--hidden-feats', type=int, default=32)
    parser.add_argument('--dropout', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--eval_steps', type=int, default=5)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    # OOM on a GPU of 16G
    assert args.device == -1, "This example can only run on cpu"
    device = 'cpu'
    device = torch.device(device)

    dataset = PygNodePropPredDataset(name='ogbn-proteins')
    data = dataset[0]
    y_true = data.y.to(device)
    node_feats = torch.ones((y_true.shape[0], 1)).to(device)
    edge_index = data.edge_index.to(device)
    edge_weights = data.edge_attr.to(device)
    split_idx = dataset.get_idx_split()
    train_idx = split_idx['train'].to(device)

    adjs = []
    for t in range(edge_weights.shape[-1]):
        adjs.append(SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weights[:, t]))

    model = RGCN(num_layers=args.num_layers,
                 in_feats=node_feats.shape[-1],
                 hidden_feats=args.hidden_feats,
                 out_feats=y_true.shape[-1],
                 num_relations=edge_weights.shape[-1],
                 dropout=args.dropout).to(device)
    evaluator = Evaluator(name='ogbn-proteins')
    logger = Logger(args.runs, args)

    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(model, node_feats, adjs,
                         y_true, train_idx, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))

            if not args.eval:
                continue

            if epoch % args.eval_steps == 0:
                result = test(model, node_feats, adjs,
                              y_true, split_idx, evaluator)
                logger.add_result(run, result)

                if epoch % args.log_steps == 0:
                    train_rocauc, valid_rocauc, test_rocauc = result
                    print(f'Run: {run + 1:02d}, '
                          f'Epoch: {epoch:02d}, '
                          f'Loss: {loss:.4f}, '
                          f'Train: {100 * train_rocauc:.2f}%, '
                          f'Valid: {100 * valid_rocauc:.2f}% '
                          f'Test: {100 * test_rocauc:.2f}%')
        if args.eval:
            logger.print_statistics(run)
    if args.eval:
        logger.print_statistics()

if __name__ == '__main__':
    main()
