import numpy as np
from ALS import ALS
import pylab as plt

# Create instance of ALS class with given problem
dims = [4,5,6,4]
T = np.random.randn(*dims)
ranks = [2,2,2,2]
ranks = [11,11,11,11]
als = ALS(T, ranks, n_epochs=100)

# Test the init cores function
als.init_cores()
for core in als.cores:
    print(core.shape)

# Test the compute subchain function
T = als.compute_subchain(0)
print(T.shape)

# Test the recover function
R = als.recover()
print(R.shape)
print(np.linalg.norm(R))
print(als.unfold(T, 0).shape)
print(np.reshape(als.T, [als.T.shape[0], -1]).shape)

# Solve a simple problem
als.solve(verbose=True, penalty='proximal', lamb=0.001)

# Solve for multiple ranks, penalties and lambdas and plot error graphs
lambdas = [0,0.001,0.01,0.1,1]
penalties = ['proximal', 'l2', None]
dims = [4,5,6,4]
ranks_list = [[2,2,2,2], [4,4,4,4], [8,8,8,8], [11,11,11,11]]
T = np.random.randn(*dims)

for r in ranks_list:
    als = ALS(T, r, n_epochs=100)
    for p in penalties:
        for l in lambdas:
            als.solve(verbose=True, penalty= p, lamb=l)
            plt.plot(range(len(als.errors)),als.errors)
            plt.yscale('log')
            plt.ylabel('Error')
            plt.xlabel('Number of iterations')
            plt.title(f'Training error with regularization={p}')
        plt.legend(lambdas)
        plt.savefig(f'losses_{p}_rank{r[0]}.png')
        plt.clf()
