# Part A
import numpy as np
import matplotlib.pyplot as plt

data1 = np.genfromtxt('/Users/kiranajith/Documents/UMD/673/Workspace/pc1.csv', delimiter=',')
data2 = np.genfromtxt('/Users/kiranajith/Documents/UMD/673/Workspace/pc2.csv', delimiter=',')

# Standard Least square method

# for fitting a surface using data from data1.csv
A = np.hstack((data1[:, :2], np.ones((data1.shape[0], 1))))
b = data1[:, 2]
x = np.linalg.inv(A.T @ A) @ A.T @ b
a, b, c = x
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_c = np.linspace(data1[:, 0].min(), data1[:, 0].max(), 100)
y_c = np.linspace(data1[:, 1].min(), data1[:, 1].max(), 100)
x1, y1 = np.meshgrid(x_c, y_c)
z1 = a*x1 + b*y1 + c
print(f'Equation of surface: {a}x + {b}y + {c} =z')
ax.plot_surface(x1, y1, z1)

plt.show()


# for fitting a surface using data from data2.csv
A = np.hstack((data2[:, :2], np.ones((data2.shape[0], 1))))
b = data2[:, 2]
x = np.linalg.inv(A.T @ A) @ A.T @ b
a, b, c = x
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x_c = np.linspace(data2[:, 0].min(), data2[:, 0].max(), 100)
y_c = np.linspace(data2[:, 1].min(), data2[:, 1].max(), 100)
x1, y1 = np.meshgrid(x_c, y_c)
z1 = a*x1 + b*y1 + c
print(f'Equation of surface: {a}x + {b}y + {c} =z')
ax.plot_surface(x1, y1, z1)

plt.show()


# Total Least square method

data_cen = data1 - np.mean(data1, axis=0)
U, D, Vt = np.linalg.svd(data_cen)
a, b, c = Vt[-1, :]

# Normalize the coefficients
norm = np.sqrt(a**2 + b**2 + 1)
a /= norm
b /= norm
c /= norm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_c = np.linspace(data1[:, 0].min(), data1[:, 0].max(), 100)
y_c = np.linspace(data1[:, 1].min(), data1[:, 1].max(), 100)
x1, y1 = np.meshgrid(x_c, y_c)
z1 = a*x1 + b*y1 + c

ax.plot_surface(x1, y1, z1)

plt.show()


data_cen = data1 - np.mean(data2, axis=0)
U, D, Vt = np.linalg.svd(data_cen)
a, b, c = Vt[-1, :]

# Normalize the coefficients
norm = np.sqrt(a**2 + b**2 + 1)
a /= norm
b /= norm
c /= norm
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_c = np.linspace(data2[:, 0].min(), data2[:, 0].max(), 100)
y_c = np.linspace(data2[:, 1].min(), data2[:, 1].max(), 100)
x1, y1 = np.meshgrid(x_c, y_c)
z1 = a*x1 + b*y1 + c

ax.plot_surface(x1, y1, z1)

plt.show()
