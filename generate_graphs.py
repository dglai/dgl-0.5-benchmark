import dgl

N = [14, 16, 18]
E = [18, 20, 22]
gl = []
for n in N:
    for e in E:
        gl.append(dgl.rand_graph(2**n, 2**e))
dgl.data.utils.save_graphs('/mnt/Projects/test_small.grp', gl)
