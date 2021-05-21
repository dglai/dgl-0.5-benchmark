import main_dgl_enzymes_gcn
import main_dgl_molhiv_gcn
import main_dgl_ppa_gcn
import main_pyg_enzymes_gcn
import main_pyg_molhiv_gcn


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


def get_output(func, queue):
    from tqdm import tqdm
    from functools import partialmethod
    import sys
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)  # disable tqdm
    sys.argv.append("--eval")  # add evaluation results
    with io.StringIO() as buf, redirect_stdout(buf):
        func()
        output = buf.getvalue()
        filename = Path(inspect.getfile(func)).name
        queue.put({filename: parse_results(output)})


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    test_list = [
        main_dgl_enzymes_gcn.main,
        main_dgl_molhiv_gcn.main,
        # main_dgl_ppa_gcn.main,
        main_pyg_enzymes_gcn.main,
        main_pyg_molhiv_gcn.main
    ]
    ret_dict = {}
    for t in test_list:
        p = ctx.Process(target=get_output, args=(t, q))
        p.start()
        p.join()
        ret = q.get(block=False)
    ret_dict.update(ret)

