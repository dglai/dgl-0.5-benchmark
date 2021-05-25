FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt update && apt install -y git

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \ 
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip install torch-geometric

RUN pip install dgl-cu111

RUN pip install tabulate ogb