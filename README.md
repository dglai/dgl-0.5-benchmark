# gnn-kernel-benchmark

## use Nsight Compute to measure global load throughput and global load transactions
```
sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH /usr/local/cuda/NsightCompute-1.0/target/linux-desktop-glibc_2_11_3-glx-x64/nv-nsight-cu-cli --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum [test command]
```
`l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second` corresponds to global load throughput and `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` corresponds to global load transactions. 