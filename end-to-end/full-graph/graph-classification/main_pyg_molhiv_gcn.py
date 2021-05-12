import argparse
import numpy as np
import time
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import DataLoader
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.utils import degree

from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder,BondEncoder

from utils import Logger

cls_loss = torch.nn.BCEWithLogitsLoss()

class GCNConv(MessagePassing):
    def __init__(self,
                 in_feats,
                 out_feats):
        super(GCNConv, self).__init__()

        self.fc = nn.Linear(in_feats, out_feats, bias=False)
        self.root_emb = nn.Embedding(1, in_feats)
        self.bond_encoder = BondEncoder(in_feats)
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.fc.weight, gain=gain)
        self.root_emb.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        x = self.fc(x)
        edge_embedding = self.bond_encoder(edge_attr)

        row, col = edge_index
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

class GCN(nn.Module):
    def __init__(self,
                 emb_dim,
                 num_classes,
                 num_layers,
                 dropout):
        super(GCN, self).__init__()

        self.atom_encoder = AtomEncoder(emb_dim)
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
        self.readout = global_mean_pool
        self.graph_pred_fc = nn.Linear(emb_dim, num_classes, bias=False)

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        self.graph_pred_fc.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch

        x = self.atom_encoder(x)
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x, edge_index, edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        x = self.layers[-1](x, edge_index, edge_attr)
        return self.graph_pred_fc(self.readout(x, batch))

def train(model, device, train_loader, optimizer):
    model.train()

    train_iter = tqdm(train_loader, desc='Training')
    for i, batch in enumerate(train_iter):
        # data copy
        batch = batch.to(device)
        # train
        optimizer.zero_grad()
        out = model(batch)
        loss = cls_loss(out.float(), batch.y.float())
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
    for step, batch in enumerate(tqdm(loader, desc='Iter')):
        # data copy
        batch = batch.to(device)
        pred = model(batch)
        y_true.append(batch.y.detach().cpu())
        y_pred.append(pred.view(batch.y.shape).detach().cpu())

    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)

    return evaluator.eval({'y_true': y_true, 'y_pred': y_pred})

def main():
    parser = argparse.ArgumentParser(description='OGBN-MolHiv')
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

    dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
    split_idx = dataset.get_idx_split()

    evaluator = Evaluator(name='ogbg-molhiv')
    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.eval_batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.eval_batch_size, shuffle=False, num_workers=0)

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

            val_rocauc = test(model, device, val_loader, evaluator)[dataset.eval_metric]
            test_rocauc = test(model, device, test_loader, evaluator)[dataset.eval_metric]
            logger.add_result(run, (0.0, val_rocauc, test_rocauc))

            if epoch % args.log_steps == 0:
                print(f'Run: {run + 1:02d}, '
                      f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Valid: {val_rocauc:.2f} '
                      f'Test: {test_rocauc:.2f}')

        if args.eval:
            logger.print_statistics(run)
    if args.eval:
        logger.print_statistics()


if __name__ == '__main__':
    main()
