import dgl
import dgl.sparse
from tvm import topi
import torch as th
from utils import partition_for_spmm
from featgraph.module import gsddmm, gspmm
import argparse
import tvm
from tvm.contrib.dlpack import to_pytorch_func
import numpy as np
import time

n_cold_start = 0
num_runs = 1
num_cores = 8
cache_size = 45 * 2 ** 20
binary_op_map = {
    'add': lambda x,y : x+y,
    'sub': lambda x,y : x-y,
    'mul': lambda x,y : x*y,
    'div': lambda x,y : x/y,
    'copy_u' : lambda x,y : x,
    'copy_v' : lambda x,y : y,
}

th_dtype_mapping = {
    'float16': th.float16,
    'float32': th.float32,
    'float64': th.float64,
    'int32': th.int32,
    'int64': th.int64
}

def prod(t):
    p = 1
    for x in t:
        p *= x
    return p

def measure_time(module, f_input, ctx):
    print('measurement start')
    for _ in range(n_cold_start):
            module(*f_input)
    timer = module.time_evaluator(module.entry_name, ctx=ctx, number=num_runs)
    tcost = timer(*f_input).mean
    print('measurement end')
    return tcost

def spmm_input(binary_op, reduce_op,
               src_shape, dst_shape, out_shape,
               feat_type, id_type, th_ctx):
    use_src = binary_op != 'copy_rhs'
    use_dst = binary_op != 'copy_lhs'
    f_input = []
    print('preparing input...')
    if use_src:
        f_input.append(th.rand(src_shape, dtype=th_dtype_mapping[feat_type], device=th_ctx))
    if use_dst:
        f_input.append(th.rand(dst_shape, dtype=th_dtype_mapping[feat_type], device=th_ctx))
    if reduce_op != 'sum':
        if use_src:
            f_input.append(th.zeros(out_shape, dtype=th_dtype_mapping[id_type], device=th_ctx))
        if use_dst:
            f_input.append(th.zeros(out_shape, dtype=th_dtype_mapping[id_type], device=th_ctx))
    out = th.zeros(out_shape, dtype=th_dtype_mapping[feat_type], device=th_ctx)
    print('done')
    f_input.append(out)
    return f_input

def torch_to_tvm(inputs):
    outputs = []
    for i in inputs:
        outputs.append(tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(i)))
    return outputs

partition1d_graphs = {}


def bench_spmm(g, target, ctx, th_ctx, binary_op, reduce_op, 
                src_feat_shape, dst_feat_shape, feat_type, test_normal=True, test_partition=False):
    num_rows, num_cols = g.number_of_dst_nodes(), g.number_of_src_nodes()
    nnz = g.number_of_edges()
    gidx = g._graph
    id_type = gidx.dtype
    print(id_type)
    out_shape = dgl.sparse.infer_broadcast_shape(binary_op, src_feat_shape, dst_feat_shape)
    f_input = spmm_input(binary_op, reduce_op,
                (num_cols,) + src_feat_shape,
                (nnz,) + dst_feat_shape,
                (num_rows,) + out_shape,
                feat_type, id_type, th_ctx)
    if test_normal:
        rst = gidx.get_csc_dlpack(0)
        adj_indptr, adj_indices, edge_mapping = map(lambda x: tvm.nd.from_dlpack(x.to_dlpack()), rst)
        f = gspmm.spmm(
                binary_op, reduce_op, nnz, num_rows, num_cols,
                src_feat_shape, dst_feat_shape, out_shape,
                id_type, feat_type, use_idx=True,
                target=target
            )
        # if binary_op != 'copy_lhs':
        #     edge_mapping = th.utils.dlpack.from_dlpack(rst[2].to_dlpack())
        #     if binary_op == 'copy_rhs':
        #         f_input[0] = f_input[0][edge_mapping.long()]
        #     else:
        #         f_input[1] = f_input[1][edge_mapping.long()]
        tcost = measure_time(f, [adj_indptr, adj_indices, edge_mapping] + torch_to_tvm(f_input), ctx)
        print('binary_op: {}\treduce_op:{}\tsrc_feat_shape: {}\tdst_feat_shape: {}\telpased time: {:.3e}s'
                .format(binary_op, reduce_op, src_feat_shape, dst_feat_shape, tcost))
    if test_partition:
        P = partition_for_spmm(num_cols, nnz, 
            prod(src_feat_shape), prod(dst_feat_shape), prod(out_shape),
            4, 4, num_cores, cache_size, binary_op, reduce_op != 'sum')
        if P == 1:
            print('No partiton needed')
        else:
            num_feat_partitions = 2
            num_col_partitions = 4
            num_cols_per_partition = (num_cols + num_col_partitions - 1) // num_col_partitions
            key = (id(g), num_col_partitions)
            if key in partition1d_graphs:
                rst = partition1d_graphs[key]
            else:
                print('start partition vertice of graph in to {} segments'.format(num_col_partitions))
                start = time.time()
                rst = dgl.sparse._CAPI_DGLPartition1D(gidx, 0, num_cols_per_partition)
                print('partition finishes within {}s'.format(time.time() - start))
                partition1d_graphs[key] = rst
            adj_indptr, adj_indices, edge_mapping = map(lambda x: tvm.nd.from_dlpack(rst(x).to_dlpack()), [0,1,2])
            # if binary_op != 'copy_lhs':
            #     edge_mapping = th.utils.dlpack.from_dlpack(rst(2).to_dlpack())
            #     print(edge_mapping)
            #     if binary_op == 'copy_rhs':
            #         f_input[0] = f_input[0][edge_mapping.long()]
            #     else:
            #         f_input[1] = f_input[1][edge_mapping.long()]
            f = gspmm.spmm(
                binary_op, reduce_op, nnz, num_rows, num_cols,
                src_feat_shape, dst_feat_shape, out_shape,
                id_type, feat_type, use_idx=False,
                num_col_partitions=num_col_partitions, 
                num_feat_partitions=num_feat_partitions,
                target=target
            )
            tcost = measure_time(f, [adj_indptr, adj_indices] + torch_to_tvm(f_input), ctx)
            print('binary_op: {}\treduce_op:{}\tsrc_feat_shape: {}\tdst_feat_shape: {}\telpased time: {:.3e}s'
                    .format(binary_op, reduce_op, src_feat_shape, dst_feat_shape, tcost))

    
def bench_sddmm(g, target, ctx, th_ctx, generic=False, 
                num_row_partitions=1, num_col_partitions=1, feat_partition=False):
    print("SDDMM\n----------------------------")
    num_rows, num_cols = g.number_of_dst_nodes(), g.number_of_src_nodes()
    nnz = g.number_of_edges()
    gidx = g._graph
    id_type = gidx.dtype
    if num_row_partitions == 1 and num_col_partitions == 1:
        row, col, _ = map(lambda x: tvm.nd.from_dlpack(x.to_dlpack()), gidx.get_coo_dlpack(0))
    else:
        num_rows_per_partition = num_rows // num_row_partitions
        num_cols_per_partition = num_cols // num_col_partitions
        print('partition start')
        start = time.time()
        rst = dgl.sparse._CAPI_DGLPartition2D(gidx, 0, num_rows_per_partition, num_cols_per_partition)
        row, col, edge_id = map(lambda x: tvm.nd.from_dlpack(rst(x).to_dlpack()), [0,1,2])
        print('partition finish within {}s'.format(time.time() - start))
    def measure_time(binary_op, src_feat_shape, dst_feat_shape, 
                     feat_type, num_warmup, num_runs):
        out_shape = dgl.sparse.infer_broadcast_shape(binary_op, src_feat_shape, dst_feat_shape)
        if feat_partition:
            num_feat_partitions = topi.util.get_const_int(topi.util.prod(out_shape)) // 32
        else:
            num_feat_partitions = 1
        if generic:
            f = gsddmm.sddmm(binary_op, 0, 0, 0, 
                             src_feat_shape, dst_feat_shape, out_shape,
                             id_type, feat_type,
                             lhs_target=0, rhs_target=2, target=target)
        else:
            f = gsddmm.sddmm(binary_op, nnz, num_rows, num_cols, 
                             src_feat_shape, dst_feat_shape, out_shape,
                             id_type, feat_type, 
                             num_feat_partitions = num_feat_partitions,
                             lhs_target=0, rhs_target=2, target=target)
        # try:
        f_input = [row,  col]
        src_feat = th.rand((num_rows,) + src_feat_shape, dtype=th_dtype_mapping[feat_type], device=th_ctx)
        tvm_src_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(src_feat))
        f_input.append(tvm_src_feat)
        dst_feat = th.rand((num_cols,) + dst_feat_shape, dtype=th_dtype_mapping[feat_type], device=th_ctx)
        tvm_dst_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(dst_feat))
        f_input.append(tvm_dst_feat)
        out = th.zeros((nnz,) + out_shape, dtype=th_dtype_mapping[feat_type], device=th_ctx)
        tvm_out = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(out))
        f_input.append(tvm_out)
        # warmup
        for _ in range(num_warmup):
            f(*f_input)
        # measure average time
        timer = f.time_evaluator(f.entry_name, ctx=ctx, number=num_runs)
        tcost = timer(*f_input).mean
        print('binary_op: {}\tsrc_feat_shape: {}\tdst_feat_shape: {}\telpased time: {:.3e}s'
            .format(binary_op, src_feat_shape, dst_feat_shape, tcost))
        # except:
        #     print(out_shape, 'OOM')
    shapes = [(2 ** x,) for x in range(8)]
    for shape in shapes:
        measure_time('add', shape, shape, 'float16', n_cold_start, num_runs)
    for shape in shapes:
        measure_time('add', shape, shape, 'float32', n_cold_start, num_runs)

if __name__ == '__main__':
    import dgl.backend as F
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    args = parser.parse_args()
    g = dgl.rand_graph(2**16, 2**22)
    if args.gpu == '-1':
        target = 'llvm'
        ctx = th.device('cpu')
        tvm_ctx = tvm.cpu(0)
    else:
        target = 'cuda'
        ctx = th.device(int(args.gpu))
        tvm_ctx = tvm.gpu(int(args.gpu))
    # for dataset in ['arxiv', 'reddit', 'proteins']:
    # for dataset in ['reddit']:
    # g = get_graph(dataset)
    g = g.astype(F.int64).to(ctx)
    print(g)
    # SPMM
    binary_op = 'add'
    reduce_op = 'sum'
    feat_type = 'float32'
    shapes = [(2, 512)]
    for shape in shapes:
        bench_spmm(g, target, tvm_ctx, ctx, 
                binary_op, reduce_op, 
                shape, shape, feat_type,
                test_normal=True,
                test_partition=True)
    # SDDMM
    # bench_sddmm(g, target, tvm_ctx, ctx, generic=False, num_row_partitions=1, num_col_partitions=1, feat_partition=False)
    # bench_sddmm(g, target, tvm_ctx, ctx, generic=False, num_row_partitions=2, num_col_partitions=2, feat_partition=False)
        