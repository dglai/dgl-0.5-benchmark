# GNN System Benchmark

The repository hosts the benchmark code from the paper [*Deep Graph Library: A Graph-centric, Highly-performant Package for Graph Neural Networks*](https://arxiv.org/pdf/1909.01315.pdf). Since both frameworks have evolved quite drastically after the paper came out (when we evaluated evaluated DGL v0.5 against PyTorch-Geometric (PyG) v1.6), we made a checkpoint of the original benchmark code in the `0.5` branch. We add more benchmark suites to the master branch and evaluate them using the latest packages.

**Benchmark Packages:**
* DGL v0.6.1
* PyTorch Geometric (PyG) v1.7.0

**Hardware:** AWS EC2 p3.2xlarge instance (one NVidia V100 GPU with 16GB GPU RAM and 8 VCPUs)

**Metric:** Time in seconds to train one epoch

**Datasets:**

Node classification
| Dataset | #Nodes  | #Edges  | Density (%)  | #Node features  | #Edge features  |
|---|---|---|---|---|---|
| cora  | 2708  |  5429 | 0.074  | 1433  |  0 |
| reddit  | 232965  |  11606919 | 0.021  |  602 |  0 |
| ogbn-arxiv | 169343  | 1166243 | 0.004  | 128  |  0 |
| ogbn-product | 2449029  | 61859140  |  0.001 | 100  |  0 |

Graph classification

| Dataset | #Graphs | #Nodes  | #Edges  | Density (%)  | #Node features  | #Edge features  |
|---|---|---|---|---|---|---| 
|ogbg-molhiv | 41127 | 25.5 | 27.5 | 4.23 | 9 | 3 |

## Node classification results

| Dataset | Model  | DGL/Time(s)  | DGL/Acc(%)  | PyG/Time(s) | PyG/Acc(%) |
|---|---|---|---|---|---|
| cora | SAGE | 0.0039 | 79.24 ± 0.93 | 0.0026 | 79.99 ± 0.49 |
| cora | GAT | 0.012 | 80.83 ± 0.98 | 0.0109 | 80.38 ± 0.49 |
| reddit | SAGE | 0.3627 | 94.86 ± 0.08 | 0.4037 | 94.94 ± 0.04 |
| reddit | GAT | 0.5532 | 90.01 | OOM | OOM |
| ogbn-arxiv | SAGE | 0.0943 | 72.08 ± 0.15 | 0.0981 | 72.00 ± 0.19 |
| ogbn-arxiv | GAT | 0.0798 | 69.53 ± 0.19 | 0.181 | 69.27 ± 0.10 |
| ogbn-product | SAGE | 0.3436 | 75.95 ± 0.19 | OOM | OOM |

## Graph classification results

On `ogbn-molhiv`

| Batch Size | DGL/Time(s)  | DGL/ROCAUC  | PyG/Time(s) | PyG/ROCAUC |
|---|---|---|---|---|
| 64 | 15.089 | 76.26 | 13.517 | 76.69 |
| 128 | 8.666 | 76.36 | 7.274 | 76.73 |
| 256 | 5.166 | 76.59 | 4.586 | 76.58 |


Note: OOM means *Out-Of-Memory*
