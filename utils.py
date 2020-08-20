import torch as th
import dgl
import time
import dgl.data
# from ogb.nodeproppred import DglNodePropPredDataset
# from torch_sparse import SparseTensor

class th_op_time(object):
    def __enter__(self):
        # if th.cuda.is_available():
        #     self.start_event = th.cuda.Event(enable_timing=True)
        #     self.end_event = th.cuda.Event(enable_timing=True)
        #     self.start_event.record()
        # else:
        self.tic = time.time()
        return self

    def __exit__(self, type, value, traceback):
        # if th.cuda.is_available():
        #     self.end_event.record()
        #     th.cuda.synchronize()  # Wait for the events to be recorded!
        #     self.time = self.start_event.elapsed_time(self.end_event)
        # else:
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
    # elif dataset == 'arxiv':
    #     arxiv = DglNodePropPredDataset(name='ogbn-arxiv')
    #     return homo_to_hetero(arxiv[0][0])
    # elif dataset == 'proteins':
    #     protein = DglNodePropPredDataset(name='ogbn-proteins')
    #     return homo_to_hetero(protein[0][0])
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

# two function calculates number of partition for spmm and sddmm
# but whether partitioning is benefinicial depends on denseness and skewness of the graph
# in future release consider a machine learning model to make this decision
def partition_for_spmm(N, E, LF, RF, OF, IL, FL, NC, CS, binary_op, comp=False):
    if binary_op == 'copy_rhs':
        return 1
    if binary_op != 'copy_lhs':
        CS = CS * LF // (LF + RF)
    adj_indptr = (NC + 1) * IL
    adj_indices = NC * IL
    ufeat = N * LF * FL
    if comp:
        if binary_op in ['copy_lhs', 'copy_rhs']:
            vfeat= NC * OF * (FL + IL)
        else:
            vfeat= NC * OF * (FL + 2 * IL)
    else:
        vfeat= NC * OF * FL
    fixed_space = adj_indptr + adj_indices + vfeat
    for np in range(1, N):
        if fixed_space + (ufeat // np) < CS:
            return np

def partition_for_sddmm(M, N, LF, RF, OF, IL, FL, NC, CS, target):
    if target == 0:
        return 1
    row = NC * IL
    col = NC * IL
    ufeat = LF * FL
    vfeat = RF * FL
    rst = NC * OF * FL
    fixed_space = row + col + rst
    def mem_size(P):
        space = fixed_space
        if target == 2:
            space += ufeat * M // P
            space += vfeat * NC
        if target == 2:
            space += vfeat * N // P
            space += ufeat * NC
        return space
    for P in range(1, min(M,N)):
        if mem_size(P) < CS:
            return P