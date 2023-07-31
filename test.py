import eden
import time
import numpy as np

np.random.seed(1234)
_a = np.random.randn(1000, 1200)
_b = np.random.randn(1200, 1000)

#eden.set_threads(1)

a = eden.Matrix(_a)
b = eden.Matrix(_b)

start = time.time()
c = a @ b
end = time.time()

total = end - start
print(c)

print(f"Execution time: {total} seconds. (Eden)")

import torch

a = torch.tensor(_a, dtype=torch.float32)
b = torch.tensor(_b, dtype=torch.float32)

start = time.time()
c = a @ b
end = time.time()

total = end - start
print(c)

print(f"Execution time: {total} seconds. (PyTorch)")