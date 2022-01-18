import numpy as np
from functools import reduce

class ALS:
    def __init__(self, T, ranks, eps=1e-6, n_epochs=100, early_stopping=True):
        self.ranks = ranks
        self.T = T
        self.eps = eps
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.cores = []

    def solve(self):
        self.init_cores()
        self.errors = []

        for epoch in range(1, self.n_epochs):
            for k, d in enumerate(self.T.shape):
                print(f'k={k}')
                print(f'epoch={epoch}')
                
                # Compute the subchain
                Q = self.compute_subchain(k)
                print(f'subchain dims: {Q.shape}')

                # Matricization of subchain tensor
                Q_mat = np.reshape(Q, (reduce(lambda x, y: x*y, Q.shape[1:len(Q.shape)-1]), Q.shape[0]*Q.shape[-1]))

                # Matricization of target tensor
                T_k = list(self.T.shape)
                T_k.pop(k)
                T_mat = np.reshape(self.T, (reduce(lambda x, y: x*y, T_k), self.T.shape[k]))
                print(f'Q dims: {Q_mat.shape}')
                print(f'T dims: {T_mat.shape}')

                # Solve with least squares solver
                A = np.linalg.lstsq(Q_mat, T_mat)[0]
                A = np.reshape(A, self.cores[k].shape)
                self.cores[k] = A

                # Calculate relative error
                R = self.recover()
                error = np.linalg.norm(self.T - R)/np.linalg.norm(self.T)
                print(f'error: {error}')
                self.errors.append(error)


    def init_cores(self):
        if not self.cores:
            for k, d in enumerate(self.T.shape):
                self.cores.append(np.random.normal(0, 1, size=(self.ranks[k], d, self.ranks[(k+1)%len(self.ranks)])))

    def compute_subchain(self, k):
        Q = self.cores[(k+1)%len(self.ranks)]
        for i in range(k+2, len(self.cores)+k):
          Q = np.tensordot(Q, self.cores[i%len(self.ranks)], axes=[-1,0])
        return Q

    def recover(self):
        R = self.cores[0]
        for i in range(1, len(self.cores)-1):
            R = np.tensordot(R, self.cores[i], axes=[-1,0])
        R = np.tensordot(R, self.cores[-1], axes=([0,-1], [-1,0]))
        return R
