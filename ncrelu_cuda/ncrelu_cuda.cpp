#include <torch/extension.h>

// CUDA函数声明
at::Tensor NCReLUForwardLauncher(const at::Tensor& src,
                                 const int batch,
                                 const int channels,
                                 const int height,
                                 const int width);

// 宏定义
#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

// C++函数包装
at::Tensor ncrelu_forward_cuda(const at::Tensor input) {
  CHECK_INPUT(input);
  at::DeviceGuard guard(input.device());	
  int batch = input.size(0);
  int channels = input.size(1);
  int height = input.size(2);
  int width = input.size(3);
  return NCReLUForwardLauncher(input, batch, channels, height, width);
}

// 绑定
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("ncrelu_forward_cuda", &ncrelu_forward_cuda,
        "ncrelu forward (CUDA)");
}