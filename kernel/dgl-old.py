import dgl
import dgl.function as fn
import torch as th
import csv
from utils import th_op_time, get_graph, binary_op_dict

import argparse

n_cold_start = 2

msg_dict = {
    'add': fn.u_add_e('x', 'w', 'm'),
    'sub': fn.u_sub_e('x', 'w', 'm'),
    'mul': fn.u_mul_e('x', 'w', 'm'),
    'div': fn.u_div_e('x', 'w', 'm'),
    'copy_u': fn.copy_u('x', 'm'),
    'copy_e': fn.copy_e('w', 'm')
}

apply_edge_dict = {
    'add': fn.u_add_v('x', 'x', 'm'),
    'sub': fn.u_sub_v('x', 'x', 'm'),
    'mul': fn.u_mul_v('x', 'x', 'm'),
    'div': fn.u_div_v('x', 'x', 'm'),
    'dot': fn.u_dot_v('x', 'x', 'm'),
    'copy_u': fn.copy_u('x', 'm'),
}

reduce_dict = {
    'sum': fn.sum('m', 'y'),
    'max': fn.max('m', 'y')
}

def bench_spmm(csvfile, g, ctx, binary_op, reduce_op):
    writer = csv.writer(csvfile)
    print("SPMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
                nfeat = th.rand(g.number_of_src_nodes(), n_hid, device=ctx)
                efeat = th.rand(g.number_of_edges(), n_hid, device=ctx) if binary_op != 'copy_u' else None
                g.srcdata['x'] = nfeat
                if binary_op != 'copy_u':
                    g.edata['w'] = efeat
                accum_time = 0
                for n_times in range(10):
                    with th_op_time() as timer:
                        g.update_all(msg_dict[binary_op],
                                     reduce_dict[reduce_op])
                        out = g.dstdata.pop('y')
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('hidden size: {}, avg time: {}'.format(
                    n_hid, avg_time))
                writer.writerow([str(n_hid), str(avg_time)])
            except:
                print('hidden size: {}, OOM'.format(n_hid))
                writer.writerow([str(n_hid), 'OOM'])
            finally:
                if 'x' in g.srcdata: g.srcdata.pop('x')
                if 'w' in g.edata: g.edata.pop('w')

def bench_sddmm(csvfile, g, ctx, op):
    writer = csv.writer(csvfile)
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
                        g.apply_edges(apply_edge_dict[op])
                        out = g.edata.pop('m')
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('hidden size: {}, avg time: {}'.format(
                    n_hid, avg_time))
                writer.writerow([str(n_hid), str(avg_time)])
            except:
                print('hidden size: {}, OOM'.format(n_hid))
                writer.writerow([str(n_hid), 'OOM'])
            finally:
                if 'x' in g.srcdata: g.srcdata.pop('x')
                if 'w' in g.edata: g.dstdata.pop('x')

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--spmm-binary', type=str, default='copy_u')
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
        print(g)
        # SPMM
        with open('_'.join(['old', dataset, 'spmm', ctx_str, args.spmm_binary, args.spmm_reduce]) + '.csv', 'w') as csvfile:
            bench_spmm(csvfile, g, ctx, args.spmm_binary, args.spmm_reduce)
        # SDDMM
        if ctx_str == 'cpu': continue  # sddmm out of mem on cpu will result in termination of the program.
        with open('_'.join(['old', dataset, 'sddmm', ctx_str, args.sddmm_binary]) + '.csv', 'w') as csvfile:
            bench_sddmm(csvfile, g, ctx, args.sddmm_binary)
