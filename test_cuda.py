import torch
# from ncrelu_cuda import ncrelu_forward_cuda
import ncrelu_cuda

a = torch.randn(4, 3).cuda()
b = ncrelu_cuda.ncrelu_forward_cuda(a)
print(b)