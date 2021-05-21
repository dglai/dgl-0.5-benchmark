import os
import random
import numpy as np
from time import time

import torch
from torch.utils.data import Dataset

import dgl
import dgl.function as fn

def arg_list(labels):
    hist, indexes, inverse, counts = np.unique(
        labels, return_index=True, return_counts=True, return_inverse=True)
    li = []
    for h in hist:
        li.append(np.argwhere(inverse == h))
    return li

def get_partition_list(g, psize):
    tmp_time = time()
    print("getting adj using time{:.4f}".format(time() - tmp_time))
    print("run metis with partition size {}".format(psize))
    nd_group = dgl.transform.metis_partition_assignment(g, psize)
    print("metis finished in {} seconds.".format(time() - tmp_time))
    print("train group {}".format(len(nd_group)))
    al = arg_list(nd_group)
    return al

class ClusterIterDataset(Dataset):
    '''The partition sampler given a DGLGraph and partition number.
    The metis is used as the graph partition backend.
    '''
    def __init__(self, dn, g, psize, use_pp=True):
        """Initialize the sampler.

        Paramters
        ---------
        dn : str
            The dataset name.
        g  : DGLGraph
            The full graph of dataset
        psize: int
            The partition number
        seed_nid: np.ndarray
            The training nodes ids, used to extract the training graph
        use_pp: bool
            Whether to use precompute of AX
        """
        self.use_pp = use_pp
        self.g = g

        # precalc the aggregated features from training graph only
        if use_pp:
            self.precalc(self.g)
            print('precalculating')

        self.psize = psize
        # cache the partitions of known datasets&partition number
        if dn:
            fn = os.path.join('./datasets/', dn + '_{}.npy'.format(psize))
            if os.path.exists(fn):
                self.par_li = np.load(fn, allow_pickle=True)
            else:
                os.makedirs('./datasets/', exist_ok=True)
                self.par_li = get_partition_list(self.g, psize)
                np.save(fn, self.par_li)
        else:
            self.par_li = get_partition_list(self.g, psize)

    def precalc(self, g):
        norm = self.get_norm(g)
        g.ndata['norm'] = norm
        features = g.ndata['features']
        print("features shape, ", features.shape)
        with torch.no_grad():
            g.update_all(fn.copy_src(src='features', out='m'),
                         fn.sum(msg='m', out='features'),
                         None)
            pre_feats = g.ndata['features'] * norm
            # use graphsage embedding aggregation style
            g.ndata['features'] = torch.cat([features, pre_feats], dim=1)

    # use one side normalization
    def get_norm(self, g):
        norm = 1. / g.in_degrees().float().unsqueeze(1)
        norm[torch.isinf(norm)] = 0
        norm = norm.to(self.g.ndata['features'].device)
        return norm

    def __len__(self):
        return self.psize

    def __getitem__(self, idx):
        return self.par_li[idx]

def subgraph_collate_fn(g, batch, negs):
    nids = np.concatenate(batch).reshape(-1).astype(np.int64)    
    g1 = g.subgraph({'_U': nids})
    for k in g.node_attr_schemes().keys():
        g1.ndata[k] = g.ndata[k][nids]
    # prepare negative sampling
    src, _ = g1.all_edges()
    dst_neg = torch.randint(0, g1.ndata['feat'].size(0), (src.size()[0] * negs,),
                            dtype=torch.long, device=src.device)

    src = src.repeat_interleave(negs)
    neg_g = dgl.graph((src, dst_neg), num_nodes=g1.number_of_nodes())
    return g1, neg_g
