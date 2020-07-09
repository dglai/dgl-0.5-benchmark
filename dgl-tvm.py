import dgl
import dgl.sparse
import torch as th
from utils import th_op_time, get_graph
from featgraph.module import gsddmm, gspmm
from featgraph.util import calc_bcast, partition_csr
import argparse
import tvm
from tvm.contrib.dlpack import to_pytorch_func
import numpy as np
import util

n_cold_start = 2
num_runs = 10
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

def bench_spmm(g, target, ctx, th_ctx, generic=False):
    print("SPMM\n----------------------------")
    adj_scipy_csr = g.adjacency_matrix(scipy_fmt='csr')
    adj_indptr = adj_scipy_csr.indptr
    adj_indices = adj_scipy_csr.indices
    num_rows, num_cols = adj_scipy_csr.shape[0], adj_scipy_csr.shape[1]
    nnz = adj_indices.shape[0]
    id_type = str(adj_indptr.dtype)
    tvm_indptr = tvm.nd.array(adj_indptr, ctx=ctx)
    tvm_indices = tvm.nd.array(adj_indices, ctx=ctx)
    def measure_time(binary_op, reduce_op, 
                     src_feat_shape, dst_feat_shape, feat_type, 
                     num_warmup, num_runs, verify=False):
        use_bcast, lhs_len, rhs_len, out_len, _, lhs_off, rhs_off = \
            calc_bcast(binary_op, np.zeros((1,)+src_feat_shape), np.zeros((1,)+dst_feat_shape))
        if generic:
            f = tvm.build(gspmm.gspmm(binary_op, reduce_op, id_type, feat_type,
                                      use_bcast=use_bcast, target=target), target=target)
        else:
            f = gspmm.spmm(
                binary_op, reduce_op, nnz, num_rows, num_cols,
                lhs_len, rhs_len, out_len,
                id_type, feat_type, 
                use_bcast=use_bcast, target=target
            )
        # try:
        use_src, use_dst = True, True
        if binary_op == 'copy_u':
            use_dst = False
        elif binary_op == 'copy_e':
            use_src = False
        f_input = [tvm_indptr, tvm_indices]
        if use_src:
            src_feat = th.rand((num_cols, lhs_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
            tvm_src_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(src_feat))
            f_input.append(tvm_src_feat)
        if use_dst:
            dst_feat = th.rand((nnz, rhs_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
            tvm_dst_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(dst_feat))
            f_input.append(tvm_dst_feat)
        if use_bcast:
            tvm_lhs_off = tvm.nd.array(lhs_off.astype(id_type), ctx=ctx)
            tvm_rhs_off = tvm.nd.array(rhs_off.astype(id_type), ctx=ctx)
            f_input += [tvm_lhs_off, tvm_rhs_off]
        out = th.zeros((num_rows, out_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
        tvm_out = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(out))
        f_input.append(tvm_out)
        # verify result is correct
        if verify and binary_op == 'mul' and reduce_op == 'sum' and out_len == 1:
            f(*f_input)
            adj_scipy_csr.data = dst_feat.cpu().numpy()
            np_result = adj_scipy_csr.dot(src_feat.cpu().numpy())
            np.testing.assert_allclose(np_result, tvm_out.asnumpy(), rtol=1e-4, atol=1e-4)
        else:
            # warmup
            for _ in range(num_warmup):
                f(*f_input)
            # measure average time
            timer = f.time_evaluator(f.entry_name, ctx=ctx, number=num_runs)
            tcost = timer(*f_input).mean
            print('binary_op: {}\treduce_op:{}\tsrc_feat_shape: {}\tdst_feat_shape: {}\telpased time: {:.3e}s'
                .format(binary_op, reduce_op, src_feat_shape, dst_feat_shape, tcost))
        # except Exception as e:
        #     print(e)
    shapes = [(2 ** x,) for x in range(3)]
    for shape in shapes:
        measure_time('mul', 'sum', shape, shape, 'float32', n_cold_start, num_runs, verify=True)
    # for shape in shapes:
    #     measure_time('copy_u', 'max', shape, shape, 'float32', n_cold_start, num_runs)
    
def bench_sddmm(g, target, ctx, th_ctx, generic=False, 
                partition=True):
    print("SDDMM\n----------------------------")
    if not partition:
        adj_scipy_coo = g.adjacency_matrix(scipy_fmt='coo')
        num_rows, num_cols = adj_scipy_coo.shape[0], adj_scipy_coo.shape[1]
        adj_row_indices = adj_scipy_coo.row
        adj_col_indices = adj_scipy_coo.col
        nnz = adj_row_indices.shape[0]
        id_type = str(adj_row_indices.dtype)
    else:
        adj_scipy_csr = g.adjacency_matrix(scipy_fmt='csr')
        num_rows, num_cols = adj_scipy_csr.shape[0], adj_scipy_csr.shape[1]
        nnz = adj_scipy_csr.indices.shape[0]
        id_type = str(adj_scipy_csr.indices.dtype)
        adj_row_indices = np.zeros(shape=(nnz,), dtype=id_type)
        adj_col_indices = np.zeros(shape=(nnz,), dtype=id_type)
        edge_id = np.arange(1, 1+nnz, dtype=id_type)
        par_edge_id = np.zeros(shape=(nnz,), dtype=id_type)
        util.partition_2d(adj_scipy_csr.indptr, adj_scipy_csr.indices, edge_id,
                       adj_row_indices, adj_col_indices, par_edge_id,
                       num_rows, num_cols, 32, 32)
    tvm_row_indices = tvm.nd.array(adj_row_indices, ctx=ctx)
    tvm_col_indices = tvm.nd.array(adj_col_indices, ctx=ctx)
    def measure_time(binary_op, src_feat_shape, dst_feat_shape, 
                     feat_type, num_warmup, num_runs, verify=False):
        use_bcast, lhs_len, rhs_len, out_len, reduce_size, lhs_off, rhs_off = \
            calc_bcast(binary_op, np.zeros((1,)+src_feat_shape), np.zeros((1,)+dst_feat_shape))
        if generic:
            f = tvm.build(gsddmm.gsddmm(binary_op, id_type, feat_type, 
                                        use_bcast=use_bcast, target=target), target=target)
        else:
            f = gsddmm.sddmm(binary_op, nnz, num_rows, num_cols, 
                             lhs_len, rhs_len, out_len,
                             id_type, feat_type, 
                             reduce_size=reduce_size, target=target)
        try:
            f_input = [tvm_row_indices, tvm_col_indices]
            use_src, use_dst = True, True
            if binary_op == 'copy_u':
                use_dst = False
            elif binary_op == 'copy_v':
                use_src = False
            if use_src:
                src_feat = th.rand((num_rows, lhs_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
                tvm_src_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(src_feat))
                f_input.append(tvm_src_feat)
            if use_dst:
                dst_feat = th.rand((num_cols, rhs_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
                tvm_dst_feat = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(dst_feat))
                f_input.append(tvm_dst_feat)
            if use_bcast:
                tvm_lhs_off = tvm.nd.array(lhs_off.astype(id_type), ctx=ctx)
                tvm_rhs_off = tvm.nd.array(rhs_off.astype(id_type), ctx=ctx)
                f_input += [tvm_lhs_off, tvm_rhs_off]
            if generic and binary_op == 'dot':
                f_input.append(reduce_size)
            out = th.zeros((nnz, out_len), dtype=th_dtype_mapping[feat_type], device=th_ctx)
            tvm_out = tvm.nd.from_dlpack(th.utils.dlpack.to_dlpack(out))
            f_input.append(tvm_out)
            # verify result is correct
            if verify:
                f(*f_input)
                lhs = src_feat[adj_row_indices]
                rhs = dst_feat[adj_col_indices]
                if binary_op != 'dot':
                    np_result = binary_op_map[binary_op](lhs, rhs)
                else:
                    if reduce_size == feat_len:
                        np_result = (lhs * rhs).sum(axis=-1, keepdims=True)
                    else:
                        lhs = lhs.reshape((lhs.shape[0], feat_len // reduce_size, reduce_size))
                        rhs = rhs.reshape((rhs.shape[0], feat_len // reduce_size, reduce_size))
                        np_result = (lhs * rhs).sum(axis=-1)
                if target == 'cuda':
                    np_result = np_result.cpu()
                np.testing.assert_allclose(np_result, tvm_out.asnumpy(), rtol=1e-4, atol=1e-4)
            else:
                # warmup
                for _ in range(num_warmup):
                    f(*f_input)
                # measure average time
                timer = f.time_evaluator(f.entry_name, ctx=ctx, number=num_runs)
                tcost = timer(*f_input).mean
                print('binary_op: {}\tsrc_feat_shape: {}\tdst_feat_shape: {}\telpased time: {:.3e}s'
                    .format(binary_op, src_feat_shape, dst_feat_shape, tcost))
        except:
            print(out_len, 'OOM')
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
        target = 'llvm'
        ctx = th.device('cpu')
        tvm_ctx = tvm.cpu(0)
    else:
        target = 'cuda'
        ctx = th.device(int(args.gpu))
        tvm_ctx = tvm.gpu(int(args.gpu))
    # for dataset in ['arxiv', 'reddit', 'proteins']:
    for dataset in ['arxiv']:
        g = get_graph(dataset)
        print(g)
        # SPMM
        bench_spmm(g, target, tvm_ctx, ctx, generic=False)
        # SDDMM
        # bench_sddmm(g, target, tvm_ctx, ctx, generic=False, partition=True)
        