import dgl
import dgl.function as fn
import torch as th
from utils import th_op_time, get_graph

import argparse

n_cold_start = 2

def bench_spmm(g, ctx):
    print("SPMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                nfeat = th.rand(g.number_of_src_nodes(), n_hid, device=ctx)
#                efeat = th.rand(g.number_of_edges(), n_hid, device=ctx)
                g.srcdata['x'] = nfeat
#                g.edata['w'] = efeat
                accum_time = 0
                for n_times in range(10):
                    with th_op_time() as timer:
#                        g.update_all(fn.u_mul_e('x', 'w', 'm'),
#                                     fn.sum('m', 'y'))
                        g.update_all(fn.copy_u('x', 'm'), fn.sum('m', 'y'))
                        out = g.dstdata.pop('y')
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('hidden size: {}, avg time: {}'.format(
                    n_hid, avg_time))
            except:
                print('hidden size: {}, OOM'.format(n_hid))
            finally:
                if 'x' in g.srcdata: g.srcdata.pop('x')
                if 'w' in g.edata: g.edata.pop('w')

def bench_sddmm(g, ctx):
    print("SDDMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                ufeat = th.rand(g.number_of_src_nodes(), n_hid, device=ctx)
                vfeat = th.rand(g.number_of_dst_nodes(), n_hid, device=ctx)
                g.srcdata['x'] = ufeat
                g.dstdata['x'] = vfeat
                accum_time = 0
                for n_times in range(10):
                    with th_op_time() as timer:
                        g.apply_edges(fn.u_dot_v('x', 'x', 'm'))
                        out = g.edata.pop('m')
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('hidden size: {}, avg time: {}'.format(
                    n_hid, avg_time))
            except err:
                print('hidden size: {}, OOM'.format(n_hid))
            finally:
                if 'x' in g.srcdata: g.srcdata.pop('x')
                if 'w' in g.edata: g.dstdata.pop('x')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    args = parser.parse_args()
    if args.gpu == '-1':
        ctx = th.device('cpu')
    else:
        ctx = th.device(int(args.gpu))

    for dataset in ['arxiv', 'proteins']:
        g = get_graph(dataset)
        print(g)
        # SPMM
        bench_spmm(g, ctx)
        # SDDMM
        bench_sddmm(g, ctx)
