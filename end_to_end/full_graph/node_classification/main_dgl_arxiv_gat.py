import argparse
import dgl
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

from utils import Logger

class GAT(nn.Module):
    def __init__(self,
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

    def forward(self, g, h):
        for l in range(self.num_layers - 1):
            h = self.gat_layers[l](g, h).flatten(1)
        logits = self.gat_layers[-1](g, h).mean(1)
        return logits.log_softmax(dim=-1)

def train(model, g, feats, y_true, train_idx, optimizer):
    model.train()

    optimizer.zero_grad()
    out = model(g, feats)[train_idx]
    loss = F.nll_loss(out, y_true.squeeze(1)[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()

@torch.no_grad()
def test(model, g, feats, y_true, split_idx, evaluator):
    model.eval()

    out = model(g, feats)
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
    parser = argparse.ArgumentParser(description='OGBN-Arxiv (GAT Full-Batch)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
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
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument("--eval", action='store_true',
                        help='If not set, we will only do the training part.')
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    g, labels = dataset[0]
    feats = g.ndata['feat'].to(device)
    labels = labels.to(device)
    train_idx = split_idx['train'].to(device)

    g = dgl.to_bidirected(g)
    g = dgl.add_self_loop(g)
    g = g.int().to(device)
    print(g)

    model = GAT(num_layers=args.num_layers,
                in_feats=feats.size(-1),
                num_hidden=args.num_hidden,
                num_classes=dataset.num_classes,
                heads=[4, 4, 4],
                feat_drop=args.dropout,
                attn_drop=args.dropout).to(device)

    evaluator = Evaluator(name='ogbn-arxiv')
    logger = Logger(args.runs, args)

    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(1, 1 + args.epochs):
            t0 = time.time()
            loss = train(model, g, feats, labels, train_idx, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))

            if not args.eval:
                continue

            result = test(model, g, feats, labels, split_idx, evaluator)
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


if __name__ == '__main__':
    main()
