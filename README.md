# GNN System Benchmark

The repository hosts the benchmark code from the paper [*Deep Graph Library: A Graph-centric, Highly-performant Package for Graph Neural Networks*](https://arxiv.org/pdf/1909.01315.pdf). Since both frameworks have evolved quite drastically after the paper came out (when we evaluated DGL v0.5 against PyTorch-Geometric (PyG) v1.6), we made a checkpoint of the original benchmark code in the `0.5` branch. We have added more benchmark suites to the master branch and evaluated them using the latest packages.

**Benchmark Packages:**
* DGL v0.6.1
* PyTorch Geometric (PyG) v1.7.0

All using Python 3.7.5, PyTorch 1.8.1 and CUDA 11.1.

**Hardware:**
* AWS EC2 p3.2xlarge instance (one NVidia V100 GPU with 16GB GPU RAM and 8 VCPUs)

**Metric:** Time in seconds to train one epoch

**Datasets:**

Node classification
| Dataset | #Nodes  | #Edges  | Density (%)  | #Node features  | #Edge features  |
|---|---|---|---|---|---|
| cora  | 2708  |  5429 | 0.074  | 1433  |  0 |
| pubmed | 19717 | 44338 | 0.011 | 500 | 0 |
| reddit  | 232965  |  11606919 | 0.021  |  602 |  0 |
| ogbn-arxiv | 169343  | 1166243 | 0.004  | 128  |  0 |
| ogbn-product | 2449029  | 61859140  |  0.001 | 100  |  0 |

Graph classification

| Dataset | #Graphs | #Nodes  | #Edges  | Density (%)  | #Node features  | #Edge features  |
|---|---|---|---|---|---|---| 
| ENZYMES | 600 | 32.63 | 62.14 | 5.83 | 18 | 0 |
|ogbg-molhiv | 41127 | 25.5 | 27.5 | 4.23 | 9 | 3 |

## Node classification results

| Dataset | Model  | DGL/Time(s)  | DGL/Acc(%)  | PyG/Time(s) | PyG/Acc(%) |
|---|---|---|---|---|---|
| cora | SAGE | 0.0039 | 79.24 ± 0.93 | 0.0026 | 79.99 ± 0.49 |
| cora | GAT | 0.012 | 80.83 ± 0.98 | 0.0109 | 80.38 ± 0.49 |
| pubmed | SAGE | 0.0046 | 77.29 ± 0.66 | 0.0031 |  77.12 ± 0.39 |
| pubmed | GAT | 0.0136 | 77.11 ± 0.93 | 0.0126 | 76.80 ± 1.21 |
| reddit | SAGE | 0.3627 | 94.86 ± 0.08 | 0.4037 | 94.94 ± 0.04 |
| reddit | GAT | 0.5532 | 89.14 ± 2.42 | OOM | OOM |
| ogbn-arxiv | SAGE | 0.0943 | 72.08 ± 0.15 | 0.0981 | 72.00 ± 0.19 |
| ogbn-arxiv | GAT | 0.0798 | 69.53 ± 0.19 | 0.181 | 69.27 ± 0.10 |
| ogbn-product | SAGE | 0.3436 | 75.95 ± 0.19 | OOM | OOM |

* Run the scripts under `end-to-end/full-graph/node-classification` with default configs to get the above results.
* Run with the `--eval` option to get model accuracy.

## Graph classification results

On `ENZYMES`

| Batch Size | DGL/Time(s)  | DGL/Acc(%)  | PyG/Time(s) | PyG/Acc(%) |
|---|---|---|---|---|
| 64 | 0.092  | 64.50 ± 3.34 | 0.081 | 63.17 ± 3.96 |
| 128 |  0.052 | 62.33 ± 3.62| 0.052 | 68.67 ± 6.28 |
| 256 | 0.039 | 63.00 ± 4.83 | 0.045 | 55.00 ± 3.69 |

On `ogbg-molhiv`

| Batch Size | DGL/Time(s)  | DGL/ROCAUC  | PyG/Time(s) | PyG/ROCAUC |
|---|---|---|---|---|
| 64 | 15.089 | 0.7666 ± 0.0125 | 13.517 | 0.7681 ± 0.0065 |
| 128 | 8.666 | 0.7615 ± 0.0139 | 7.274 | 0.7690 ± 0.0109 |
| 256 | 5.166 | 0.7694 ± 0.0080 | 4.586 | 0.7756 ± 0.0106 |

* Run the scripts under `end-to-end/full-graph/graph-classification` with default configs to get the above results.
* Run with the `--eval` option to get model accuracy.
* Run with the `--batch_size` option to set the training batch size.
* DGL is ~10% slower on `ogbg-molhiv` majorly because the graphs have nearly 1:1 node-to-edge ratio, where PyG's scatter-gather kernel becomes slightly better than DGL's fused message passing kernels.

## Notes
* Since the purpose is to evaluate system efficiency, we focus on matching the implementations in different frameworks instead of on getting the best model performance.
* Both PyG and DGL provide APIs at different levels. It is not fair to compare higher-level APIs like NN modules with lower-level ones such as message passing. This benchmark suite choose lower-level APIs as the attempt to compare the best speed one can get from the frameworks.
* We implement the PyG benchmarks using its recommended APIs whenever possible. Please create a github issue if you feel some implementation can be improved.
* OOM means *Out-Of-Memory*
