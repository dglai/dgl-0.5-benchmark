import dgl
import dgl.ops
import torch as th
from utils import th_op_time, get_graph

import argparse

n_cold_start = 2

def bench_spmm(g, ctx, binary_op, reduce_op):
    print("SPMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                nfeat = th.rand(g.number_of_src_nodes(), n_hid, device=ctx)
                efeat = th.rand(g.number_of_edges(), n_hid, device=ctx) if binary_op != 'copy_lhs' else None
                accum_time = 0
                for n_times in range(10):
                    with th_op_time() as timer:
                        dgl.ops.gspmm(g, binary_op, reduce_op, nfeat, efeat)
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('hidden size: {}, avg time: {}'.format(
                    n_hid, avg_time))
            except:
                print('hidden size: {}, OOM'.format(n_hid))

def bench_sddmm(g, ctx, op):
    print("SDDMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                ufeat = th.rand(g.number_of_src_nodes(), n_hid, device=ctx)
                vfeat = th.rand(g.number_of_dst_nodes(), n_hid, device=ctx)
                accum_time = 0
                for n_times in range(10):
                    with th_op_time() as timer:
                        dgl.ops.gsddmm(g, op, ufeat, vfeat)
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('hidden size: {}, avg time: {}'.format(
                    n_hid, avg_time))
            except:
                print('hidden size: {}, OOM'.format(n_hid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--spmm-binary', type=str, default='copy_lhs')
    parser.add_argument('--spmm-reduce', type=str, default='sum')
    parser.add_argument('--sddmm-binary', type=str, default='add')
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    args = parser.parse_args()
    if args.gpu == '-1':
        ctx = th.device('cpu')
    else:
        ctx = th.device(int(args.gpu))
    ctx_str = 'cpu' if args.gpu == '-1' else 'gpu'

    for dataset in ['reddit', 'arxiv', 'proteins']:
        g = get_graph(dataset)
        g = g.int().to(ctx)
        print(g)
        # SPMM
        bench_spmm(g, ctx, args.spmm_binary, args.spmm_reduce)
        # SDDMM
        if ctx_str == 'cpu': continue  # sddmm out of mem on cpu will result in termination of the program.
        bench_sddmm(g, ctx, args.sddmm_binary)
        del g
