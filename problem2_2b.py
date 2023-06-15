# Part b 
import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt('/Users/kiranajith/Documents/UMD/673/Workspace/pc1.csv', delimiter=',')

def ransac(data, n, k, t, d):
    b_m = None
    b_e = np.inf
    for i in range(k):
        indices = np.random.choice(data.shape[0], n, replace=False)
        p = data[indices]
        nrl = np.cross(p[1] - p[0], p[2] - p[0])
        d = -np.sum(nrl * p[0])
        md = np.concatenate((nrl, [d]))

        error = np.abs(np.dot(data, md[:-1]) + md[-1]) / np.linalg.norm(md[:-1])

        inls = np.sum(error < t)
        outls = data.shape[0] - inls

        if inls > d and inls < b_e:
            b_m = md
            b_e = inls

        if inls >= d:
            break

    return b_m

md = ransac(data, n=3, k=100, t=0.01, d=100)

xx, yy = np.meshgrid(np.arange(-0.5, 0.5, 0.01), np.arange(-0.5, 0.5, 0.01))
z = (-md[0] * xx - md[1] * yy - md[3]) / md[2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='b', marker='o')
ax.plot_surface(xx, yy, z, alpha=0.5)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()
