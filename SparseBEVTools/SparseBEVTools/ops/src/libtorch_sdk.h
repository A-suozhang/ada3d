#ifndef LIBTORCH_SDK_H
#define LIBTORCH_SDK_H
#include <torch/serialize/tensor.h>
#include <vector>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <iostream>
#include <torch/extension.h>
#define THREADS_PER_BLOCK 128

using namespace std;

#define CHECK_CUDA(x) do { \
  if (!x.device().is_cuda()) { \
    fprintf(stderr, "%s must be CUDA tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)
#define CHECK_CONTIGUOUS(x) do { \
  if (!x.is_contiguous()) { \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n", #x, __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)

#define CHECK_DIM(x, num) do { \
  if (x.dim() != num) { \
    fprintf(stderr, "%s should have %d dims but find %d at %s:%d\n", #x, num, x.dim(), __FILE__, __LINE__); \
    exit(-1); \
  } \
} while (0)

#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor>
sparse_to_bev_sum_along_z(torch::Tensor features, torch::Tensor coors, vector<int> spatial_shape, int batch_size);
torch::Tensor
sparse_to_bev_sum_along_z_backward(torch::Tensor grad, torch::Tensor coors);

void sparse_to_bev_sum_along_z_launcher(const float* features_ptr, const int* coors_ptr,
                                        float* bev_dense_ptr, int* valid_mask_ptr,
                                        const int num_voxels, const int batch_size,
                                        const int channels, std::vector<int>& spatial_shape);
void sparse_to_bev_sum_along_z_backward_launcher(const float* grad_ptr, const int* coors_ptr, float* output_grad_ptr,
                                                const int num_voxels, const int batch_size, const int channels,
                                                const int H, const int W);


std::vector<torch::Tensor>
foreground_gather(torch::Tensor features, torch::Tensor coors, torch::Tensor valid_mask,
                  std::vector<int> spatial_shape, int batch_size, int max_voxels);
torch::Tensor
foreground_gather_backward(torch::Tensor grad, torch::Tensor out_idx_map, int num_in_voxels);

void foreground_gather_launcher(const float* features_ptr, const int* coors_ptr, const int* valid_mask_ptr,
                                float* out_features_ptr, int* out_coors_ptr, int* idx_map_ptr, int* count_ptr,
                                const int num_voxels, const int batch_size,
                                const int channels, const int max_voxels, std::vector<int>& spatial_shape);
void foreground_gather_backward_launcher(const float* grad_ptr, const int* out_idx_map_ptr, float* out_grad_ptr,
                                         const int num_in_voxels, const int num_out_voxels, const int channels);

void points_in_boxes_bev_launcher(const float* point_cloud_ptr, const float* boxes_ptr, int* out_mask_ptr,
                                const int num_points, const int num_point_attr, const int num_boxes, const int num_box_attr);
#endif