import numpy as np
from ALS import ALS
dims = [4,5,6,4]
T = np.random.randn(*dims)
ranks = [11,11,11,11]
als = ALS(T, ranks)

als.init_cores()
for core in als.cores:
    print(core.shape)

T = als.compute_subchain(0)
print(T.shape)

R = als.recover()
print(R.shape)

print(als.unfold(T, 0).shape)
print(np.reshape(als.T, [als.T.shape[0], -1]).shape)

als.solve()
als.plot_losses()
