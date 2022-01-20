import numpy as np
from ALS import ALS
ranks = [2,3,4,5]
T = np.arange(72)
T = np.reshape(T, (4,3,2,3))
als = ALS(T, ranks)

als.init_cores()
for core in als.cores:
    print(core.shape)

T = als.compute_subchain(0)
print(T.shape)

R = als.recover()
print(R.shape)
als.solve()
