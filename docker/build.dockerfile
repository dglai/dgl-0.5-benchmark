FROM pytorch/pytorch:1.8.0-cuda11.1-cudnn8-devel

RUN apt update && apt install -y git && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    sudo ./aws/install

RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \ 
    pip install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html && \
    pip install torch-geometric

RUN pip install dgl-cu111

RUN pip install tabulate ogb pytest nose numpy cython scipy networkx matplotlib nltk requests[security] tqdm