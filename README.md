# TR-ALS: A Tensor Ring Decomposition using Alternating Least Squares
`tr-als` is a solver for decomposing arbitrary order tensors into their tensor ring factorization using alternating least squares.


## Description

This solver was written in the context of the honours project (IFT3150) at University of Montréal. The goal of the project is to analyze and compare the performance of the ALS algorithm to the greedy algorithm  for the specific task of tensor ring decomposition [(Hashemizadeh et al., 2018)](https://arxiv.org/abs/2008.05437).

### Dependencies

In order to install the required dependencies, navigate to the project repository and run the following command:
```
pip install -r requirements.txt
```

### Executing program

In order to execute the solver, first start by creating an instance of the class, then call the `solve` method. More details about the parameters for this method are given in the docstrings.
```
als = ALS(T, ranks, n_epochs=100)
als.solve(verbose=True)

```
`T` is the tensor to factorize, `ranks` is a list of integers corresponding to the ranks and `n_epochs` is the number of iterations the solver will run for. For more information on the parameters, see the docstrings in the code.

## Authors

Michael Rizvi-Martel
[michael.rizvi-martel@umontreal.ca](michael.rizvi-martel@umontreal.ca)

## Acknowledgments

Thanks to my supervisor [Guillaume Rabusseau](https://www-labs.iro.umontreal.ca/~grabus/) for his knowledge about tensors and all the help given while debugging the code!
