import argparse
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.dataloading import GraphDataLoader
from dgl.data import LegacyTUDataset
from dgl.nn.pytorch import GraphConv as GCNConv
import dgl.function as fn

from utils import Logger

class GCN(nn.Module):
    def __init__(self,
                 feat_size,
                 hidden_size,
                 num_classes,
                 num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        # input layer
        self.convs.append(GCNConv(feat_size, hidden_size, bias=False, allow_zero_in_degree=True))
        self.bns.append(nn.BatchNorm1d(hidden_size))
        # hidden convs
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_size, hidden_size, bias=False, allow_zero_in_degree=True))
            self.bns.append(nn.BatchNorm1d(hidden_size))
        # output layer
        self.convs.append(GCNConv(hidden_size, hidden_size, bias=False, allow_zero_in_degree=True))
        self.dropout = nn.Dropout(p=dropout)
        self.readout = dgl.nn.AvgPooling()
        self.graph_fcs = nn.ModuleList()
        self.graph_fcs.append(nn.Linear(hidden_size, hidden_size))
        self.graph_fcs.append(nn.Linear(hidden_size, num_classes))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for fc in self.graph_fcs:
            gain = nn.init.calculate_gain('relu')
            nn.init.xavier_uniform_(fc.weight, gain=gain)

    def forward(self, g, x):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(g, x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.convs[-1](g, x)
        xg = self.readout(g, x)
        for i, fc in enumerate(self.graph_fcs[:-1]):
            xg = fc(xg)
            xg = F.relu(xg)
            xg = self.dropout(xg)
        xg = self.graph_fcs[-1](xg)
        return F.log_softmax(xg, dim=-1)

def train(model, device, train_loader, optimizer):
    model.train()

    train_iter = tqdm(train_loader, desc='Training')
    for i, (batched_graph, labels) in enumerate(train_iter):
        # data copy
        batched_graph = batched_graph.to(device).formats('coo')
        labels = labels.to(device)
        # train
        optimizer.zero_grad()
        x = batched_graph.ndata['feat']
        out = model(batched_graph, x)
        loss = F.nll_loss(out, labels)
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            train_iter.set_description(f'Training Loss {loss.item():.4f}', refresh=True)

    return loss.item()

@torch.no_grad()
def test(model, device, loader):
    model.eval()

    y_true = []
    y_pred = []
    for step, (batched_graph, labels) in enumerate(tqdm(loader, desc='Iter')):
        # data copy
        batched_graph = batched_graph.to(device).int()
        labels = labels.to(device)
        x = batched_graph.ndata['feat']
        pred = model(batched_graph, x)
        y_true.append(labels.detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    _, indices = torch.max(y_pred, dim=1)
    correct = torch.sum(indices == y_true)
    return correct.item() * 1.0 / len(y_true)

def main():
    parser = argparse.ArgumentParser(description='ENZYMES')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--eval', action='store_true',
                        help='If not set, we will only do the training part.')
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = LegacyTUDataset('ENZYMES')
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.seed(42)
    np.random.shuffle(indices)

    train_set = dgl.data.utils.Subset(dataset, indices[:int(num_samples * 0.8)])
    val_set = dgl.data.utils.Subset(dataset, indices[int(num_samples * 0.8):int(num_samples * 0.9)])
    test_set = dgl.data.utils.Subset(dataset, indices[int(num_samples * 0.9):int(num_samples)])

    train_loader = GraphDataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = GraphDataLoader(val_set, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)
    test_loader = GraphDataLoader(test_set, batch_size=args.eval_batch_size, shuffle=True, num_workers=0)

    model = GCN(18,
                args.hidden_size,
                num_classes=int(dataset.num_labels),
                num_layers=args.num_layers,
                dropout=args.dropout).to(device)

    logger = Logger(args.runs, args)
    dur = []
    for run in range(args.runs):
        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            loss = train(model, device, train_loader, optimizer)
            if epoch >= 3:
                dur.append(time.time() - t0)
                print('Training time/epoch {}'.format(np.mean(dur)))

            if not args.eval:
                continue

            val_acc = test(model, device, val_loader)
            test_acc = test(model, device, test_loader)
            logger.add_result(run, (0.0, val_acc, test_acc))

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Valid: {val_acc * 100:.4f}% '
                      f'Test: {test_acc * 100:.4f}%')

        if args.eval:
            logger.print_statistics(run)
    if args.eval:
        logger.print_statistics()


if __name__ == '__main__':
    main()
