import time

import torch


tensor = torch.randn(30000, 5, 5, 128, device="cuda")
tensor2 = torch.randn(128, 128, device="cuda")


torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    tensor @ tensor2
torch.cuda.synchronize()
end = time.time()
print("Time taken: ", end - start)


torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    tensor.reshape(30000, 25, 128).swapaxes(1, 2).reshape(30000, 128, 5, 5).contiguous()
torch.cuda.synchronize()
end = time.time()
print("Time taken: ", end - start)
