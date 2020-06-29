import torch as th
from torch_sparse import matmul, SparseTensor
from utils import th_op_time, get_pyg_graph
import argparse

n_cold_start = 2

def bench_spmm(g, ctx):
    adj_t = g[1].to(ctx)
    print("SPMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                nfeat = th.rand(adj_t.size(1), n_hid, device=ctx)
                accum_time = 0
                for n_times in range(10):
                    with th_op_time() as timer:
                        out = matmul(adj_t, nfeat, reduce='max')
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('hidden size: {}, avg time: {}'.format(
                    n_hid, avg_time))
            except:
                print('hidden size: {}, OOM'.format(n_hid))

def bench_sddmm(g, ctx):
    eidx = g[0]
    """
    print("SDDMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8]:
            ufeat = th.rand(g.number_of_src_nodes(), n_hid, device=ctx)
            vfeat = th.rand(g.number_of_dst_nodes(), n_hid, device=ctx)
            accum_time = 0
            for n_times in range(10):
                with th_op_time() as timer:
                    dgl.sparse.gsddmm(g, 'dot', ufeat, vfeat)
                if n_times >= n_cold_start:
                    accum_time += timer.time
            avg_time = accum_time / (n_times - n_cold_start)
            print('hidden size: {}, avg time: {}'.format(
                n_hid, avg_time))
    """
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    args = parser.parse_args()
    if args.gpu == '-1':
        ctx = th.device('cpu')
    else:
        ctx = th.device(int(args.gpu))

    for dataset in ['arxiv', 'proteins']:
        g = get_pyg_graph(dataset)
        print(dataset)
        # SPMM
        bench_spmm(g, ctx)
        # SDDMM
        bench_sddmm(g, ctx)
