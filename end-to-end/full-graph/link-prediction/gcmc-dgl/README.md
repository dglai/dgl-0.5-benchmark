# Graph Convolutional Matrix Completion

ml-100k, no feature
```bash
python3 train.py --data_name=ml-100k --use_one_hot_fea --gcn_agg_accum=stack
```

ml-1m, no feature
```bash
python3 train.py --data_name=ml-1m --gcn_agg_accum=sum --use_one_hot_fea
```

ml-10m, no feature
```bash
python3 train.py --data_name=ml-10m --gcn_agg_accum=stack --gcn_dropout=0.3 \
                                 --train_lr=0.001 --train_min_lr=0.0001 --train_max_iter=15000 \
                                 --use_one_hot_fea --gen_r_num_basis_func=4
```
