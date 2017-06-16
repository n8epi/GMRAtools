import numpy as np
import numpy.random as rd
import matplotlib.pyplot as plt

from covertree.covertree import CoverTree


def l2(p, q):
    return np.sqrt((p - q) @ (p - q))


def extract_node_data(n0, x, y, levels, links):
    for k in n0.children.keys():
        for n in n0.children[k]:
            x.append(n.data[0])
            y.append(n.data[1])
            levels.append(k)
            links.append(n0.data)
            links.append(n.data)
            extract_node_data(n, x, y, levels, links)


num_pts = 100
x = np.reshape(rd.randn(2 * num_pts), (num_pts, 2))
x = np.apply_along_axis(lambda v: v / np.sqrt(np.sum(v ** 2)), 1, x)
cube_ct = CoverTree(l2)
for i in range(num_pts):
    cube_ct.insert(x[i, :])

x = []
y = []
r = {}
levels = []
links = []

x.append(cube_ct.root.data[0])
y.append(cube_ct.root.data[1])
levels.append(cube_ct.maxlevel)

r['min'] = cube_ct.minlevel
r['max'] = cube_ct.maxlevel

extract_node_data(cube_ct.root, x, y, levels, links)

h = (np.array(levels) > -3)
x = np.array(x)
y = np.array(y)

plt.figure(1)
plt.scatter(x, y)
plt.scatter(x[h], y[h])
plt.axis('square')
plt.show()

