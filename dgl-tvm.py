import dgl
import dgl.sparse
import torch as th
from utils import th_op_time, get_graph
from featgraph.module import gsddmm
from featgraph.util import calc_bcast
import argparse
import tvm
from tvm.contrib.dlpack import to_pytorch_func
import numpy as np

n_cold_start = 5
num_runs = 20
binary_op_map = {
    'add': lambda x,y : x+y,
    'sub': lambda x,y : x-y,
    'mul': lambda x,y : x*y,
    'div': lambda x,y : x/y,
    'copy_u' : lambda x,y : x,
    'copy_v' : lambda x,y : y,
}

th_dtype_mapping = {
    'float32': th.float32,
    'float64': th.float64
}

def bench_spmm(g, ctx):
    print("SPMM\n----------------------------")
    pass


def bench_sddmm(g, f, ctx, th_ctx):
    print("SDDMM\n----------------------------")
    adj_scipy_coo = g.adjacency_matrix(scipy_fmt='coo')
    adj_row_indices = adj_scipy_coo.row
    adj_col_indices = adj_scipy_coo.col
    id_type = str(adj_row_indices.dtype)
    tvm_row_indices = tvm.nd.array(adj_row_indices, ctx=ctx)
    tvm_col_indices = tvm.nd.array(adj_col_indices, ctx=ctx)
    def measure_time(binary_op, src_feat_shape, dst_feat_shape, feat_type, num_warmup, num_runs, verify=False):
        use_bcast, lhs_len, rhs_len, out_len, reduce_size, lhs_off, rhs_off = \
            calc_bcast(binary_op, np.zeros((1,)+src_feat_shape), np.zeros((1,)+dst_feat_shape))
        f_name = 'sddmm_{}_{}_{}'.format(binary_op, id_type, feat_type)
        if use_bcast:
            f_name += '_bcast'
        src_feat = th.rand((adj_scipy_coo.shape[0], lhs_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
        dst_feat = th.rand((adj_scipy_coo.shape[1], rhs_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
        tvm_src_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(src_feat))
        tvm_dst_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(dst_feat))
        f_input = [tvm_row_indices, tvm_col_indices, tvm_src_feat, tvm_dst_feat]
        if use_bcast:
            tvm_lhs_off = tvm.nd.array(lhs_off.astype(id_type), ctx=ctx)
            tvm_rhs_off = tvm.nd.array(rhs_off.astype(id_type), ctx=ctx)
            f_input += [tvm_lhs_off, tvm_rhs_off]
        if binary_op == 'dot':
            f_input.append(reduce_size)
        out = th.zeros((adj_row_indices.shape[0], out_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
        tvm_out = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(out))
        f_input.append(tvm_out)
        # verify result is correct
        if verify:
            f.get_function(f_name)(*f_input)
            lhs = src_feat.reshape((src_feat.shape[0],)+src_feat_shape)[adj_scipy_coo.row]
            rhs = dst_feat.reshape((dst_feat.shape[0],)+dst_feat_shape)[adj_scipy_coo.col]
            if binary_op != 'dot':
                np_result = binary_op_map[binary_op](lhs, rhs)
            else:
                np_result = (lhs * rhs).sum(axis=-1)
            if th_ctx.type == 'cuda':
                np_result = np_result.cpu()
            np.testing.assert_allclose(np_result.reshape(np_result.shape[0], out_len), tvm_out.asnumpy(), rtol=1e-4, atol=1e-4)
        # warmup
        for _ in range(num_warmup):
            f.get_function(f_name)(*f_input)
        # measure average time
        timer = f.time_evaluator(f_name, ctx=ctx, number=num_runs)
        tcost = timer(*f_input).mean
        print('binary_op: {}\tsrc_feat_shape: {}\tdst_feat_shape: {}\telpased time: {:.3e}s'
              .format(binary_op, src_feat_shape, dst_feat_shape, tcost))
        # return tcost
    # lhs_shapes = [(1, 3), (5, 3), (1, 3, 3)]
    # rhs_shapes = [(5, 1), (1, 3), (3, 1, 3)]
    # for lhs_shape, rhs_shape in zip(lhs_shapes, rhs_shapes):
    #     measure_time('add', lhs_shape, rhs_shape, 'float32', n_cold_start, num_runs)
    #     if lhs_shape[-1] == rhs_shape[-1]:
    #         measure_time('dot', lhs_shape, rhs_shape, 'float32', n_cold_start, num_runs)
    # flatten_lhs_shapes = [(15,), (27,)]
    # flatten_rhs_shapes = [(15,), (27,)]
    # for lhs_shape, rhs_shape in zip(flatten_lhs_shapes, flatten_rhs_shapes):
    #     measure_time('add', lhs_shape, rhs_shape, 'float32', n_cold_start, num_runs)
    # dot_lhs_shapes = [(5, 3), (9, 3)]
    # dot_rhs_shapes = [(5, 3), (9, 3)]
    # for lhs_shape, rhs_shape in zip(dot_lhs_shapes, dot_rhs_shapes):
    #     measure_time('dot', lhs_shape, rhs_shape, 'float32', n_cold_start, num_runs)
    shapes = [(2 ** x,) for x in range(8)]
    for shape in shapes:
        measure_time('add', shape, shape, 'float32', n_cold_start, num_runs)
    # for shape in shapes:
    #     measure_time('dot', shape, shape, 'float32', n_cold_start, num_runs)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Benchmark DGL kernels")
    parser.add_argument('--gpu', '-g', type=str, default='-1')
    args = parser.parse_args()
    if args.gpu == '-1':
        ctx = th.device('cpu')
        tvm_ctx = tvm.cpu(0)
        module = gsddmm.build_all('llvm')
    else:
        ctx = th.device(int(args.gpu))
        tvm_ctx = tvm.gpu(int(args.gpu))
        module = gsddmm.build_all('cuda')
        # print(module.imported_modules[0].get_source())
    for dataset in ['proteins']:
        g = get_graph(dataset)
        print(g)
        # SPMM
        # bench_spmm(g, ctx)
        # SDDMM
        bench_sddmm(g, module, tvm_ctx, ctx)