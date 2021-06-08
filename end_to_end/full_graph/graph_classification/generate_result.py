import main_dgl_enzymes_gcn
import main_dgl_molhiv_gcn
import main_dgl_ppa_gcn
import main_pyg_enzymes_gcn
import main_pyg_molhiv_gcn


import pandas as pd
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
        main_dgl_enzymes_gcn.main,
        main_dgl_molhiv_gcn.main,
        main_dgl_ppa_gcn.main,
        main_pyg_enzymes_gcn.main,
        main_pyg_molhiv_gcn.main,
        # main_pyg_proteins_rgcn_for.main, # skip this one since will OOM
    ]
    ret_dict = {}
    for t in test_list:
        filename = Path(inspect.getfile(t)).name
        extra_argv = []
        print(f"Run {filename}.main test")
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

