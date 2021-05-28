import pandas as pd
import json
import main_dgl_arxiv_gat
import main_dgl_arxiv_sage
import main_dgl_citation_gat
import main_dgl_citation_sage
import main_dgl_product_sage
import main_dgl_proteins_rgcn_for
import main_dgl_reddit_gat
import main_dgl_reddit_sage
import main_pyg_arxiv_gat
import main_pyg_arxiv_sage
import main_pyg_citation_gat
import main_pyg_citation_sage
import main_pyg_product_sage
import main_pyg_proteins_rgcn_for


import torch
import numpy as np
import torch.multiprocessing as mp

import io
from contextlib import redirect_stdout
import inspect
from pathlib import Path


def parse_results(output: str):
    lines = output.split("\n")
    epoch_times = []
    final_train_acc = ""
    final_test_acc = ""
    for line in lines:
        line = line.strip()
        if line.startswith("Training time/epoch"):
            epoch_times.append(float(line.split(' ')[-1]))
        if line.startswith("Final Train"):
            final_train_acc = line.split(":")[-1]
        if line.startswith("Final Test"):
            final_test_acc = line.split(":")[-1]
    return {"epoch_time": np.array(epoch_times)[-10:].mean(),
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc}


def get_output(func, queue, extra_args = []):
    from tqdm import tqdm
    from functools import partialmethod
    import sys, os
    dirpath = Path(os.path.join(os.getcwd(), 'dataset'))
    if not dirpath.exists():
        os.symlink('/tmp/dataset/', os.path.join(os.getcwd(), 'dataset'))
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # disable tqdm
    sys.argv = sys.argv[:1] # clear previous argv
    sys.argv.append("--eval")  # add evaluation results
    sys.argv.extend(extra_args)  # add evaluation results
    print(sys.argv)
    with io.StringIO() as buf, redirect_stdout(buf):
        func()
        output = buf.getvalue()
        filename = Path(inspect.getfile(func)).stem
        queue.put({filename: parse_results(output)})


if __name__ == "__main__":
    print(f"CUDA device count: {torch.cuda.device_count()}")
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    test_list = [
        main_dgl_arxiv_gat.main,
        main_dgl_arxiv_sage.main,
        main_dgl_citation_gat.main,
        main_dgl_citation_sage.main,
        main_dgl_product_sage.main,
        main_dgl_proteins_rgcn_for.main,
        main_dgl_reddit_gat.main,
        main_dgl_reddit_sage.main,
        main_pyg_arxiv_gat.main,
        main_pyg_arxiv_sage.main,
        main_pyg_citation_gat.main,
        main_pyg_citation_sage.main,
        main_pyg_product_sage.main,
        # main_pyg_proteins_rgcn_for.main, # skip this one since will OOM
    ]
    ret_dict = {}
    for t in test_list:
        filename = Path(inspect.getfile(t)).name
        extra_argv = []
        print(f"Run {filename}.main test")
        if "citation" in filename:
            for citation_dataset_name in ["cora", "pubmed"]:
                extra_argv=["--dataset", citation_dataset_name]
                p = ctx.Process(target=get_output, args=(t, q, extra_argv))
                print(f"Run {t.__name__} test with argv: {extra_argv}")
                p.start()
                p.join()
                if q.empty():
                    print(f"Failed to run {filename} with argv: {extra_argv}")
                else:
                    ret = q.get(block=False)
                    fname = list(ret.keys())[0]
                    ret[f"{fname}_{citation_dataset_name}"] = ret[fname]
                    del ret[fname]
                    ret_dict.update(ret)
        else:
            p = ctx.Process(target=get_output, args=(t, q))
            p.start()
            p.join()
            if q.empty():
                print(f"Failed to run {filename}")
            else:
                ret = q.get(block=False)
                ret_dict.update(ret)
    print(ret_dict)
    df = pd.DataFrame(ret_dict).transpose()
    print(df.to_json())
    print(df.to_markdown())
    print(df)
