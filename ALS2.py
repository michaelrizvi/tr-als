import numpy as np
from functools import reduce
import matplotlib.pyplot as plt

class ALS:
    def __init__(self, T, ranks, eps=1e-6, n_epochs=50, early_stopping=True):
        self.ranks = ranks
        self.T = T
        self.eps = eps
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.cores = []

    def solve(self, verbose=True):
        self.init_cores()
        self.errors = []

        for epoch in range(1, self.n_epochs):

            for k, d in enumerate(self.T.shape):
                # k = 1
                # Compute the subchain and reshape de subchain
                Q = self.compute_subchain(k)    # R2 x d3 x d4 x d1 x R1
                Q = np.moveaxis(Q, 0, -1)  # d3 x d4 x d1 x R1 x R2
                Q_mat = np.reshape(Q, (np.prod(Q.shape[:-2]), Q.shape[-2]*Q.shape[-1])) 
                # d3d4d1 x R1R2

                # Matricization of target tensor
                T_perm = self.T
                for i in range(k):
                   T_perm = np.moveaxis(T_perm,0,-1)  # d2 x d3 x d4 x d1

                T_mat = self.unfold(T_perm, 0).T  #  d3d4d1 x d2

                # Solve with least squares solver
                A_mat = np.linalg.lstsq(Q_mat, T_mat, rcond=None)[0]    #R1R2 x d2
                shape = self.cores[k].shape
                A = np.reshape(A_mat, (shape[0], shape[2], shape[1]))  # R1 x R2 x d2
                A = np.moveaxis(A, 1, -1)  # R1 x d2 x R2 
                print("**", shape)
                print("**", A.shape)
                print("**", A.shape)
                self.cores[k] = A
                #A = np.transpose(A,[0,2,1])
                # Calculate relative error
                R = self.recover()
                error = np.linalg.norm(self.T - R)
                self.errors.append(error)

                # Calculate least squares errors
                lstsq_error = np.linalg.norm(T_mat - Q_mat.dot(A_mat))

                if verbose:
                    print(f'epoch={epoch}')
                    print(f'k={k}')
                    print(f'subchain dims: {Q.shape}')
                    print(f'A dims: {A.shape}')
                    print(f'A_mat dims: {A_mat.shape}')
                    print(f'Q dims: {Q_mat.shape}')
                    print(f'T dims: {T_mat.shape}')
                    print(f'error: {error}')
                    print(f'least squares error: {lstsq_error}')
                    print("mat diff norm",np.linalg.norm(T_mat - Q_mat.dot(A_mat) - self.unfold(self.T-R,k).T))
                    print("T", np.linalg.norm(self.T))
                    print("R", np.linalg.norm(R))
                    print("\n")

    def init_cores(self):
        if not self.cores:
            for k, d in enumerate(self.T.shape):
                self.cores.append(np.random.normal(0, 0.1, size=(self.ranks[k], d, self.ranks[(k+1)%len(self.ranks)])))

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

    def unfold(self, T, mode):
        shape = T.shape
        T = np.moveaxis(T,mode,0)
        T = T.reshape(shape[mode],-1)
        return T

    def fold3d(self, A, mode):
        shape = self.cores[mode].shape
        A = np.reshape(A, (shape[0], shape[2], shape[1]))
        A = np.transpose(A,[0,2,1])
        return A

    def plot_losses(self):
        fig, ax = plt.subplots(1)
        ax.plot(self.errors, '.k')
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title('Log loss of training')
        plt.show()
