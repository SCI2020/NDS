#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <THC/THCAtomics.cuh>
#include <stdio.h>

#define THREADS_PER_BLOCK 1024
#define PI 3.141592653589793

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

// template <typename scalar_t>
// __global__ void NCReLUForward(const int input_size,
//                               const int channels,
//                               const int height,
//                               const int width,
//                               const scalar_t * src_data,
//                               scalar_t * dst_data) {

//   const int index = blockIdx.x * blockDim.x + threadIdx.x;          // 计算绝对索引
//   // stdout << index;
//   if (index > input_size) return;
//   auto value = src_data[index];                                              // 寻找到原数据值
//   const int chw = channels * height * width;
//   dst_data[index + index / chw * chw] = value >= 0 ? value : scalar_t(0);             // 前一半通道为正值
//   dst_data[index + index / chw * chw + chw] = value >= 0 ? scalar_t(0) : value;    // 后一半通道为负值
// } 

// template <typename scalar_t>
// __global__ void NCReLUForward(const scalar_t * camera_grid_positions,
//                               const scalar_t * r,
//                               scalar_t * dst_data) {

//   const int index = blockIdx.x * blockDim.x + threadIdx.x;
//   // printf("%d %d %d\n", blockDim.x, blockDim.y, threadIdx.x);
//   printf("%d %d\n", blockIdx.x, threadIdx.x);
//   dst_data[index] = camera_grid_positions[index];
//   // dst_data[0][index] = camera_grid_positions[0][index];
//   // dst_data[1][index] = camera_grid_positions[1][index];
//   // dst_data[2][index] = camera_grid_positions[2][index];

//   // dst_data[0, index] = camera_grid_positions[0, index];
//   // dst_data[1, index] = camera_grid_positions[1, index];
//   // dst_data[2, index] = camera_grid_positions[2, index];
// } 

// template <typename scalar_t>
// __global__ void NCReLUForward(const scalar_t * camera_grid_positions,
//                               const scalar_t * r,
//                               scalar_t * dst_data,
//                               const int num_sampling_points
//                               ) {

//   // const int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
//   // const int index = threadId_2D + (blockDim.x*blockDim.y) * blockIdx.x;  
//   int threadId = threadIdx.x + threadIdx.y*blockDim.x + threadIdx.z*blockDim.x*blockDim.y;
//   int index = threadId + (blockDim.x*blockDim.y*blockDim.z)*blockIdx.x;

//   const scalar_t theta = PI / (num_sampling_points-1) * threadIdx.y;
//   const scalar_t phi = PI / (num_sampling_points-1) * threadIdx.z;
//   if (threadIdx.x == 1){
//     // const scalar_t theta = PI / (num_sampling_points-1) * threadIdx.x;
//     dst_data[index] = theta;
//   }
//   if (threadIdx.x == 0){
//     // const scalar_t phi = PI / (num_sampling_points-1) * threadIdx.y;
//     dst_data[index] = phi;
//   }  
  
//   // const scalar_t theta = PI / num_sampling_points * threadIdx.x;
//   // const scalar_t phi = PI / num_sampling_points * threadIdx.y;
//   // printf("%d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
//   printf("%d %d %d %d %f %f\n", threadIdx.x, threadIdx.y, threadIdx.z, index, theta, phi);
//   // printf("%d %d %d %d %d %d %d\n", blockDim.x, blockDim.y, threadId, index, threadIdx.x, threadIdx.y, threadIdx.z);
  
//   // dst_data[index] = camera_grid_positions[index];

// } 

// template <typename scalar_t>
// __global__ void NCReLUForward(const scalar_t * camera_grid_positions,
//                               const scalar_t * r,
//                               scalar_t * dst_data,
//                               const int num_sampling_points
//                               ) {

//   // const int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
//   // const int index = threadId_2D + (blockDim.x*blockDim.y) * blockIdx.x;  
//   int threadId = threadIdx.x + threadIdx.y*blockDim.x;
//   int index = threadId + (blockDim.x*blockDim.y)*blockIdx.x;  //

//   const scalar_t theta = PI / (num_sampling_points-1) * threadIdx.x;
//   const scalar_t phi = PI / (num_sampling_points-1) * threadIdx.y;
//   if (blockIdx.x == 0){
//     // const scalar_t theta = PI / (num_sampling_points-1) * threadIdx.x;
//     dst_data[index] = theta;
//   }
//   if (blockIdx.x == 1){
//     // const scalar_t phi = PI / (num_sampling_points-1) * threadIdx.y;
//     dst_data[index] = phi;
//   }  
  
//   // const scalar_t theta = PI / num_sampling_points * threadIdx.x;
//   // const scalar_t phi = PI / num_sampling_points * threadIdx.y;
//   // printf("%d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
//   printf("%d %d %d %d %d %f %f\n", blockIdx.x, threadIdx.x, threadIdx.y, threadIdx.z, index, theta, phi);
//   // printf("%d %d %d %d %d %d %d\n", blockDim.x, blockDim.y, threadId, index, threadIdx.x, threadIdx.y, threadIdx.z);
  
//   // dst_data[index] = camera_grid_positions[index];

// } 

template <typename scalar_t>
__global__ void NCReLUForward(const scalar_t * camera_grid_positions,
                              const scalar_t * r,
                              scalar_t * dst_data,
                              scalar_t * dst_dir_data,
                              const int num_sampling_points,
                              const int batch
                              ) {

  // const int threadId_2D = threadIdx.x + threadIdx.y * blockDim.x;
  // const int index = threadId_2D + (blockDim.x*blockDim.y) * blockIdx.x;  
  int threadId_2D = threadIdx.x + threadIdx.y*blockDim.x;
  int blockId_2D = blockIdx.x + blockIdx.y*gridDim.x;
  int index = threadId_2D + (blockDim.x*blockDim.y)*blockId_2D;

  const scalar_t theta = PI / (num_sampling_points-1) * threadIdx.y;
  const scalar_t phi = PI / (num_sampling_points-1) * threadIdx.x;
  if (blockIdx.y == 0){
    // const scalar_t theta = PI / (num_sampling_points-1) * threadIdx.x;
    // dst_data[index] = theta;
    // dst_data[index] = r[blockIdx.x];
    // dst_data[index] = r[blockIdx.x] * sin(theta) * cos(phi);
    dst_data[index] = r[blockIdx.x] * sin(theta) * cos(phi) + camera_grid_positions[blockIdx.x];
    dst_dir_data[index] = sin(theta) * cos(phi);
  }
  if (blockIdx.y == 1){
    // const scalar_t phi = PI / (num_sampling_points-1) * threadIdx.y;
    // dst_data[index] = theta;
    // dst_data[index] = phi;
    // dst_data[index] = r[blockIdx.x] * sin(theta) * sin(phi);
    // printf("%d %f %f %f\n", blockIdx.x, camera_grid_positions[blockIdx.x*3], camera_grid_positions[blockIdx.x*3+1], camera_grid_positions[blockIdx.x*3+2]);
    dst_data[index] = r[blockIdx.x] * sin(theta) * sin(phi) + camera_grid_positions[batch+blockIdx.x];
    dst_dir_data[index] = sin(theta) * sin(phi);
  }  
  if (blockIdx.y == 2){
    // const scalar_t phi = PI / (num_sampling_points-1) * threadIdx.y;
    // dst_data[index] = phi;
    // dst_data[index] = r[blockIdx.x];
    // dst_data[index] = r[blockIdx.x] * cos(theta);  
    dst_data[index] = r[blockIdx.x] * cos(theta) + camera_grid_positions[batch*2+blockIdx.x]; 
    dst_dir_data[index] = cos(theta);
  }    
  // const scalar_t theta = PI / num_sampling_points * threadIdx.x;
  // const scalar_t phi = PI / num_sampling_points * threadIdx.y;
  // printf("%d %d %d\n", blockDim.x, blockDim.y, blockDim.z);
  // printf("%d %d %d %d %d %d %f %f\n", blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, threadIdx.z, index, theta, phi);
  // printf("%d %d %d %d %d %d %d\n", blockDim.x, blockDim.y, threadId, index, threadIdx.x, threadIdx.y, threadIdx.z);
  
  // dst_data[index] = camera_grid_positions[index];
} 

// at::Tensor NCReLUForwardLauncher(const at::Tensor& src,
//                                  const int batch,
//                                  const int channels,
//                                  const int height,
//                                  const int width) {

//   at::Tensor dst = at::empty({batch, 2 * channels, height, width},    // 开辟一段存储空间
//                              src.options());
//   const int input_size = batch * channels * height * width;
//   const int output_size = batch * channels * height * width;
//   AT_DISPATCH_FLOATING_TYPES_AND_HALF(src.scalar_type(), "NCReLUForwardLauncher", ([&] {
//         const scalar_t *src_ = src.data<scalar_t>();
//         scalar_t *dst_ = dst.data<scalar_t>();

//         NCReLUForward<scalar_t>
//             <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK,
//             0, at::cuda::getCurrentCUDAStream()>>>(
//                input_size, channels, height, width, src_, dst_
//             );
//       }));
// //   THCudaCheck(cudaGetLastError());
//   return dst;
// }

std::vector<at::Tensor> NCReLUForwardLauncher(const at::Tensor& camera_grid_positions,
                                 const at::Tensor& r,
                                 const int num_sampling_points,
                                 const int batch
                                 ) {

  // at::Tensor dst = at::empty({3, batch},    // 开辟一段存储空间
  //                            camera_grid_positions.options());
  at::Tensor dst = at::empty({3, batch, num_sampling_points, num_sampling_points},    // 开辟一段存储空间
                             camera_grid_positions.options());
  at::Tensor dst_dir = at::empty({3, batch, num_sampling_points, num_sampling_points},    // 开辟一段存储空间
                             camera_grid_positions.options());
  // printf("%f %f %f\n", camera_grid_positions[0, 3], camera_grid_positions[1, 3], camera_grid_positions[2, 3]);
  // print(camera_grid_positions);
  // printf("\n");

  // const int threads = num_sampling_points * num_sampling_points;
  const dim3 threads(num_sampling_points, num_sampling_points);
  const dim3 blocks(batch, 3);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(camera_grid_positions.scalar_type(), "NCReLUForwardLauncher", ([&] {
        const scalar_t *camera_grid_positions_ = camera_grid_positions.data<scalar_t>();
        const scalar_t *r_ = r.data<scalar_t>();
        scalar_t *dst_ = dst.data<scalar_t>();
        scalar_t *dst_dir_ = dst_dir.data<scalar_t>();
        // print(camera_grid_positions_.size());
        // printf("%d \n", camera_grid_positions_.size(1));
        // printf("\n");
        NCReLUForward<scalar_t>
            <<<blocks, threads>>>(
               camera_grid_positions_, r_, dst_, dst_dir_, num_sampling_points, batch
            );
      }));
//   THCudaCheck(cudaGetLastError());
  return {dst, dst_dir};
}
