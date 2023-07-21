import eden
import time

eden.set_threads(1)

start = time.time()

mod_a = eden.Tensor([23, 12, 53])
mod_b = eden.Tensor([78, 21, 243])

a = eden.Tensor([1, 2, 3], requires_grad=True) * mod_a
b = eden.Tensor([4, 5, 6], requires_grad=True) * mod_a / mod_b
c = eden.Tensor([7, 8, 9], requires_grad=True) * mod_b
d = eden.Tensor([10, 11, 12], requires_grad=True)
e = (a @ b) + (c @ d)
e.backward()

end = time.time()

duration = end - start
print('Eden execution time:', duration, 'seconds')

print(a, a.grad)
print(b, b.grad)
print(c, c.grad)
print(d, d.grad)
print(e)

import torch
import time

start = time.time()

mod_a = torch.Tensor([23, 12, 53]); mod_a.requires_grad=True
mod_b = torch.Tensor([78, 21, 243]); mod_b.requires_grad=True

a = torch.Tensor([1, 2, 3]); a.requires_grad=True
a = a * mod_a
b = torch.Tensor([4, 5, 6]); b.requires_grad=True
b = b * mod_a / mod_b
c = torch.Tensor([7, 8, 9]); c.requires_grad=True
c = c * mod_b
d = torch.Tensor([10, 11, 12]); d.requires_grad=True
e = (a @ b) + (c @ d)
e.backward()

end = time.time()

duration = end - start
print('Eden execution time:', duration, 'seconds')

#print(a, a.grad)
#print(b, b.grad)
#print(c, c.grad)
print(d, d.grad)
print(e)