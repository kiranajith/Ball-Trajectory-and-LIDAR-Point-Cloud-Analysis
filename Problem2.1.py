import numpy as np

# reading the data
data = np.loadtxt('/Users/kiranajith/Documents/UMD/673/Workspace/pc1.csv', delimiter=',')

# finding the mean
x_mean = np.mean(data[:,0])
y_mean = np.mean(data[:,1])
z_mean = np.mean(data[:,2])

# finding the centered data
centered = data - np.array([x_mean, y_mean, z_mean])

# product of centered data matrix with its transpose
cov_matrix = np.dot(centered.T, centered)

# obtaining covariance matrix
cov_matrix /= (data.shape[0] - 1)

print("Covariance matrix:\n",cov_matrix)

# Find eigenvectors and eigenvalues of covariance matrix
eig_val, eig_vec = np.linalg.eig(cov_matrix)

# to find the eigen vector corresponding to the smallest eigen value
max_to_min = np.argsort(eig_val)[::-1]
eig_val = eig_val[max_to_min]
eig_vec = eig_vec[:,max_to_min]

dir = eig_vec[:,2]
mag = np.sqrt(eig_val[2])

print("Direction of the Surface normal:", dir)
print("Magnitude of the Suraface normal:", mag)
