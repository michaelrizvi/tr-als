import numpy as np
import matplotlib.pyplot as plt
import wandb

class ALS:
    def __init__(self, T, ranks, eps=1e-6, n_epochs=50, early_stopping=True, use_wandb=False):
        """Short summary.

        Parameters
        ----------
        T : numpy ndarray
            The target tensor to decompose.
        ranks : list
            list of the ranks for the tensors in tensor ring.
        eps : floating point number
            Error value for early stopping.
        n_epochs : int
            Number of epochs of ALS to run before stopping.
        early_stopping : boolean
            Toggles the early stopping parameter.

        Returns
        -------
        None

        """
        self.ranks = ranks
        self.T = T
        self.eps = eps
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.cores = []
        self.use_wandb = use_wandb

    def solve(self, verbose=True, penalty=None, lamb=0.001, cvg_threshold =1e-5):
        """Main method. Solves the TR decomposition using alternating least squares

        Parameters
        ----------
        verbose : boolean
            Toggle the priting of information to the console.
        penalty : string
            The type of regularization to be used by the solver.
            Allowed values are 'l2' for l2 regularization and 'proximal' for proximal regularization.
        lamb : type
            Description of parameter `lamb`.

        Returns
        -------
        type
            Description of returned object.

        """
        # TODO: RETURN THE LOSSES
        """Short summary.

        Parameters
        ----------
        verbose : boolean
            Toggle the printing of information to the consol.

        Returns
        -------
        None
           Nothing is returned, the decomposition of the target tensor T is stored in the cores attribute.

        """
        self.penalty = penalty
        self.init_cores()
        self.errors = []
        cores_prev = self.cores
        prev_loss = np.inf
        for epoch in range(1, self.n_epochs):
            for k, d in enumerate(self.T.shape):
                # Compute the subchain and reshape de subchain
                Q = self.compute_subchain(k)    # R2 x d3 x d4 x d1 x R1
                Q = np.moveaxis(Q, 0, -1)  # d3 x d4 x d1 x R1 x R2
                Q_mat = np.reshape(Q, (np.prod(Q.shape[:-2]), Q.shape[-2]*Q.shape[-1])) # d3d4d1 x R1R2

                # Matricization of target tensor
                T_perm = self.T
                for i in range(k):
                   T_perm = np.moveaxis(T_perm,0,-1)  # d2 x d3 x d4 x d1
                T_mat = self.unfold(T_perm, 0).T  #  d3d4d1 x d2

                # Solve least squares problem depending on chosen regularization
                if penalty == 'l2':
                    A_mat = np.linalg.lstsq(np.vstack((Q_mat, lamb*np.eye(Q_mat.shape[1]))),
                                            np.vstack((T_mat, np.zeros((Q_mat.shape[1], T_mat.shape[1])))), rcond=None)[0]

                elif penalty == 'proximal':
                    A_hat = self.unfold(cores_prev[k], 1).T
                    A_mat = np.linalg.lstsq(np.vstack((Q_mat, lamb*np.eye(Q_mat.shape[1]))),
                                            np.vstack((T_mat, lamb*A_hat)), rcond=None)[0]
                elif penalty == None:
                    A_mat = np.linalg.lstsq(Q_mat, T_mat, rcond=None)[0]    #R1R2 x d2

                # Update cores_prev for proximal regularization
                cores_prev = self.cores

                # Fold the resulting matrix into 3D tensor and update cores
                A = self.fold3d(A_mat, k)
                self.cores[k] = A

                # Calculate relative error
                R = self.recover()
                error = np.linalg.norm(self.T - R)
                self.errors.append(error)

                if self.use_wandb:
                    wandb.log({"loss":error})


                if verbose:
                    lstsq_error = np.linalg.norm(T_mat - Q_mat.dot(A_mat))
                    print(f'epoch={epoch}')
                    print(f'k={k}')
                    print(f'subchain dims: {Q.shape}')
                    print(f'A dims: {A.shape}')
                    print(f'A_mat dims: {A_mat.shape}')
                    print(f'Q dims: {Q_mat.shape}')
                    print(f'T dims: {T_mat.shape}')
                    print(f'error: {error}')
                    print(f'least squares error: {lstsq_error}')
                    print(f'diff: {np.abs(prev_loss - error)}')
                    #   print("\n")

            if np.abs(prev_loss - error) < cvg_threshold:
                return

            prev_loss = error

    def init_cores(self):
        """Initializes the core tensors based on the given rank values.

        Returns
        -------
        None

        """
        self.cores = []
        for k, d in enumerate(self.T.shape):
            self.cores.append(np.random.normal(0, 0.1, size=(self.ranks[k], d, self.ranks[(k+1)%len(self.ranks)])))

    def compute_subchain(self, k):
        """Contracts all the tensors in cores except for the one at kth index.

        Parameters
        ----------
        k : int
            index of the tensor to ignore.

        Returns
        -------
        ndarray
            The result of the tensor contraction.
        """
        Q = self.cores[(k+1)%len(self.ranks)]
        for i in range(k+2, len(self.cores)+k):
          Q = np.tensordot(Q, self.cores[i%len(self.ranks)], axes=[-1,0])
        return Q

    def recover(self):
        """Recovers the approximation of target tensor by contracting all tensors in cores.

        Returns
        -------
        ndarray
            The resulting tensor.

        """
        R = self.cores[0]
        for i in range(1, len(self.cores)-1):
            R = np.tensordot(R, self.cores[i], axes=[-1,0])
        R = np.tensordot(R, self.cores[-1], axes=([0,-1], [-1,0]))
        return R

    def unfold(self, T, mode):
        """Unfolds (matricizes) a tensor along the given mode.

        Parameters
        ----------
        T : ndarray
            The tensor to unfold.
        mode : integer
            The mode for the unfolding.

        Returns
        -------
        ndarray
            The matricization of T along the given mode.

        """
        shape = T.shape
        T = np.moveaxis(T,mode,0)
        T = T.reshape(shape[mode],-1)
        return T

    def fold3d(self, A, k):
        """Helper function to fold a matrix into a 3D tensor along a given mode.

        Parameters
        ----------
        A : ndarray
            The matrix to fold.
        mode : int
            mode to fold along.

        Returns
        -------
        ndarray
            The folded tensor.

        """
        shape = self.cores[k].shape
        A = np.reshape(A, (shape[0], shape[2], shape[1]))  # R1 x R2 x d2
        A = np.moveaxis(A, 1, -1)  # R1 x d2 x R2
        return A

    def plot_losses(self):
        """Plots the losses as a function of epochs.

        Returns
        -------
        None

        """
        fig, ax = plt.subplots(1)
        ax.plot(self.errors, '.k')
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.title(f'Log loss of training with {self.penalty} regularization')
        #plt.savefig(f'losses_{self.penalty}.png')
        plt.show()

if __name__ == '__main__':
    target = np.random.randn(3,4,5,6)
    ranks = [3,3,3,3]
    als = ALS(target, ranks, n_epochs=100)
    als.solve(verbose=True, penalty='proximal', lamb=0.001)
