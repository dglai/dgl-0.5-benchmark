import argparse
import dgl
import dgl.function as fn
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.dataloading import GraphDataLoader
from ogb.graphproppred import DglGraphPropPredDataset, Evaluator

from utils import Logger

cls_loss = torch.nn.CrossEntropyLoss()

class GCNConv(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(GCNConv, self).__init__()

        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        self.root_emb = nn.Embedding(1, in_feats)
        self.edge_encoder = nn.Linear(7, out_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        nn.init.xavier_uniform_(self.edge_encoder.weight, gain=gain)
        self.root_emb.reset_parameters()

    def forward(self, graph, feat, edge_feat):
        graph = graph.local_var()
 
        x = self.fc(feat)
        deg = graph.in_degrees().float().unsqueeze(1) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        graph.ndata['c'] = deg_inv_sqrt
        graph.ndata['x'] = x
        graph.edata['w'] = self.edge_encoder(edge_feat)
        graph.update_all(self.message, fn.sum('m', 'h'))
        h = graph.ndata['h']
        return h + F.relu(x + self.root_emb.weight) * 1./deg

    def message(self, edges):
        norm = edges.src['c'] * edges.dst['c']
        return {'m' : norm * F.relu(edges.src['x'] + edges.data['w'])}

class GCN(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_classes,
                 num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.node_encoder = nn.Embedding(1, emb_dim)
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        # input layer
        self.layers.append(GCNConv(emb_dim, emb_dim))
        self.bns.append(nn.BatchNorm1d(emb_dim))
        # hidden layers
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(emb_dim, emb_dim))
            self.bns.append(nn.BatchNorm1d(emb_dim))
        # output layer
        self.layers.append(GCNConv(emb_dim, emb_dim))
        self.dropout = nn.Dropout(p=dropout)
        self.readout = dgl.nn.AvgPooling()
        self.graph_pred_fc = nn.Linear(emb_dim, num_classes, bias=False)

    def reset_parameters(self):
        self.node_encoder.reset_parameters()
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.graph_pred_fc.reset_parameters()

    def forward(self, g, x, edge_attr):
        x = self.node_encoder(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(g, x, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](g, x, edge_attr)
        return self.graph_pred_fc(self.readout(g, x))

def train(model, device, train_loader, optimizer):
    model.train()

    train_iter = tqdm(train_loader, desc='Training')
    for i, (batched_graph, labels) in enumerate(train_iter):
        # data copy
        batched_graph = batched_graph.to(device).int().formats('coo')
        labels = labels.to(device)
        # train
        optimizer.zero_grad()
        nfeat = batched_graph.ndata['feat']
        efeat = batched_graph.edata['feat']
        out = model(batched_graph, nfeat, efeat)
        loss = cls_loss(out.float(), labels.float())
        loss.backward()
        optimizer.step()

        if i % 10 == 0:
            train_iter.set_description(f'Training Loss {loss.item():.4f}', refresh=True)

    return loss.item()

@torch.no_grad()
def test(model, device, loader, evaluator):
    model.eval()

    y_true = []
    y_pred = []
    for step, (batched_graph, labels) in enumerate(tqdm(loader, desc='Iter')):
        # data copy
        batched_graph = batched_graph.to(device).int()
        labels = labels.to(device)
        nfeat = batched_graph.ndata['feat']
        efeat = batched_graph.edata['feat']
        pred = model(batched_graph, nfeat, efeat)
        y_true.append(labels.detach().cpu())
        y_pred.append(pred.view(labels.shape).detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

def main():
    parser = argparse.ArgumentParser(description='OGBN-PPA')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=5)
    parser.add_argument('--emb_dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--eval', action='store_true',
                        help='If not set, we will only do the training part.')
    parser.add_argument('--eval_batch_size', type=int, default=2048)
    args = parser.parse_args()
    print(args)

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    dataset = DglGraphPropPredDataset(name='ogbg-ppa')
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(name='ogbg-ppa')
    train_loader = GraphDataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = GraphDataLoader(dataset[split_idx["valid"]], batch_size=args.eval_batch_size, shuffle=True, num_workers=0)
    test_loader = GraphDataLoader(dataset[split_idx["test"]], batch_size=args.eval_batch_size, shuffle=True, num_workers=0)

    model = GCN(args.emb_dim,
                num_classes=dataset.num_tasks,
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

            val_acc = test(model, device, val_loader, evaluator)[dataset.eval_metric]
            test_acc = test(model, device, test_loader, evaluator)[dataset.eval_metric]
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
