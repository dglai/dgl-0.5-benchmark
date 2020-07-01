import torch as th
import dgl
import time
import dgl.data
from ogb.nodeproppred import DglNodePropPredDataset
# from torch_sparse import SparseTensor

class th_op_time(object):
    def __enter__(self):
        if th.cuda.is_available():
            self.start_event = th.cuda.Event(enable_timing=True)
            self.end_event = th.cuda.Event(enable_timing=True)
            self.start_event.record()
        else:
            self.tic = time.time()
        return self

    def __exit__(self, type, value, traceback):
        if th.cuda.is_available():
            self.end_event.record()
            th.cuda.synchronize()  # Wait for the events to be recorded!
            self.time = self.start_event.elapsed_time(self.end_event)
        else:
            self.time = time.time() - self.tic


def homo_to_hetero(g):
    if not isinstance(g, dgl.DGLHeteroGraph):
        return dgl.graph(g.edges())
    return g

def dgl_to_pyg_graph(g):
    eidx = g.edges()
    N = g.number_of_nodes()
    E = g.number_of_edges()
    adj_t = SparseTensor(
        row=eidx[0], col=eidx[1], value=th.ones(E).float(), sparse_sizes=(N, N)).t()
    return eidx, adj_t

def get_graph(dataset):
    if dataset == 'reddit':
        reddit = dgl.data.RedditDataset()
        return homo_to_hetero(reddit[0])
    elif dataset == 'arxiv':
        arxiv = DglNodePropPredDataset(name='ogbn-arxiv')
        return homo_to_hetero(arxiv[0][0])
    elif dataset == 'proteins':
        protein = DglNodePropPredDataset(name='ogbn-proteins')
        return homo_to_hetero(protein[0][0])
    else:
        raise KeyError("Unrecognized dataset name: {}".format(dataset))

def get_pyg_graph(dataset):
    if dataset == 'reddit':
        reddit = dgl.data.RedditDataset()
        return dgl_to_pyg_graph(reddit[0])
    elif dataset == 'arxiv':
        arxiv = DglNodePropPredDataset(name='ogbn-arxiv')
        return dgl_to_pyg_graph(arxiv[0][0])
    elif dataset == 'proteins':
        proteins = DglNodePropPredDataset(name='ogbn-proteins')
        return dgl_to_pyg_graph(proteins[0][0])
