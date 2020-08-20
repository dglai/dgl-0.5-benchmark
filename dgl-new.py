import dgl
import dgl.backend as F
import torch as th
from utils import th_op_time
import time
import argparse

n_cold_start = 2

def calc_partition(N, F, nt):
    CS = 45 * (2 ** 20)
    if not nt:
        CS //= 2
    NF = N * F * 4
    return (NF + CS - 1) // CS

def bench_spmm(g, ctx, impl):
    print("SPMM {}\n----------------------------".format(impl))
    with th.no_grad():
        for n_hid in [8, 16, 32, 64, 128, 256, 512, 1024]:
            if impl in ['dds', 'ddsnt']:
                num_col_partitions = calc_partition(g.number_of_src_nodes(), n_hid, impl == 'ddsnt')
                if num_col_partitions == 1:
                    print('no need for partition')
                    continue
                num_cols_per_partition = (g.number_of_src_nodes() + num_col_partitions - 1) // num_col_partitions
                print('start {}-way partitioning', num_col_partitions)
                start = time.time()
                rst = dgl.sparse._CAPI_DGLPartition1D(g._graph, 0, num_cols_per_partition)
                print('partitioned, cost: {}s', time.time()-start)
            try:
                nfeat = th.rand(g.number_of_src_nodes(), n_hid, device=ctx, dtype=th.float32)
                efeat = th.rand(g.number_of_edges(), n_hid, device=ctx, dtype=th.float32)
                if impl in ['dds', 'ddsnt']:
                    out = th.zeros(g.number_of_dst_nodes(), n_hid, device=ctx, dtype=th.float32)
                    nfeat = F.zerocopy_to_dgl_ndarray(nfeat)
                    efeat = F.zerocopy_to_dgl_ndarray(efeat)
                    out = F.zerocopy_to_dgl_ndarray_for_write(out)
                accum_time = 0
                for n_times in range(10):
                    with th_op_time() as timer:
                        if impl == 'dds':
                            dgl.sparse._CAPI_DGLKernelSpMMDDS(rst(0), rst(1), nfeat, efeat, out)
                        elif impl == 'ddsnt':
                            dgl.sparse._CAPI_DGLKernelSpMMDDS_NT(rst(0), rst(1), nfeat, efeat, out)
                        else:
                            dgl.sparse._gspmm(g._graph, 'mul', 'sum', nfeat, efeat)
                        # dgl.sparse._gspmm(g, 'copy_u', 'sum', nfeat, None)
                    if n_times >= n_cold_start:
                        accum_time += timer.time
                avg_time = accum_time / (n_times - n_cold_start)
                print('{},\t{}'.format(
                    n_hid, avg_time))
            except:
                print('{},\tOOM'.format(n_hid))

def bench_sddmm(g, ctx):
    print("SDDMM\n----------------------------")
    with th.no_grad():
        for n_hid in [1, 2, 4, 8, 16, 32, 64, 128]:
            try:
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
            except:
                print('hidden size: {}, OOM'.format(n_hid))

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    parser.add_argument('--impl', '-i', type=str, default='orig')
    args = parser.parse_args()
    if args.gpu == '-1':
        ctx = th.device('cpu')
    else:
        ctx = th.device(int(args.gpu))

    # for dataset in ['arxiv', 'proteins']:
    #     g = get_graph(dataset)
    #     print(g)
    #     # SPMM
    #     bench_spmm(g, ctx)
    #     # SDDMM
    #     bench_sddmm(g, ctx)
    gl, _ = dgl.data.utils.load_graphs('/mnt/Projects/test_small.grp')
    print('done loading')
    for g in gl:
        print(g)
        bench_spmm(g, ctx, args.impl)
