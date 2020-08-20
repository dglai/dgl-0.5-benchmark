#!/bin/bash

cp ~/Projects/dgl/libdglnt.so ~/Projects/dgl/build/libdgl.so
python -u dgl-new.py -i nt | tee rst_nt.txt

cp ~/Projects/dgl/libdglntdds.so ~/Projects/dgl/build/libdgl.so
python -u dgl-new.py -i dds | tee rst_dds.txt
python -u dgl-new.py -i ddsnt | tee rst_ddsnt.txt

cp ~/Projects/dgl/libdgl.so ~/Projects/dgl/build/libdgl.so
python -u dgl-new.py -i orig | tee rst_orig.txt