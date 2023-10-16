#include "libtorch_sdk.h"

__global__ void sparse_to_bev_sum_along_z_kernel(const float* features_ptr, const int* coors_ptr,
                                        float* bev_dense_ptr, int* valid_mask_ptr,
                                        const int num_voxels, const int batch_size,
                                        const int channels, const int H, const int W,
                                        const int total_points){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total_points){
        return;
    }
    const int cur_voxel_idx = idx / channels;
    const int cur_channel_idx = idx % channels;
    const float cur_feature = features_ptr[idx];
    const int cur_batch = coors_ptr[cur_voxel_idx * 4];
    const int cur_h = coors_ptr[cur_voxel_idx * 4 + 2];
    const int cur_w = coors_ptr[cur_voxel_idx * 4 + 3];

    int offset = cur_batch * channels * H * W + cur_channel_idx * H * W + cur_h * W + cur_w;
    atomicAdd(bev_dense_ptr + offset, cur_feature);

    offset = cur_batch * H * W +  cur_h * W + cur_w;
    valid_mask_ptr[offset] = 1;
}

__global__ void sparse_to_bev_sum_along_z_backward_kernel(const float* grad_ptr, const int* coors_ptr, float* output_grad_ptr,
                                                const int num_voxels, const int batch_size, const int channels,
                                                const int H, const int W, const int total_points){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= total_points){
        return;
    }
    const int cur_voxel_idx = idx / channels;
    const int cur_channel_idx = idx % channels;
    const int cur_batch = coors_ptr[cur_voxel_idx * 4];
    const int cur_h = coors_ptr[cur_voxel_idx * 4 + 2];
    const int cur_w = coors_ptr[cur_voxel_idx * 4 + 3];

    int offset = cur_batch * channels * H * W + cur_channel_idx * H * W + cur_h * W + cur_w;
    const float cur_grad = grad_ptr[offset];
    output_grad_ptr[idx] = cur_grad;
}


__global__ void
foreground_gather_kernel(const float* features_ptr, const int* coors_ptr, const int* valid_mask_ptr,
                        float* out_features_ptr, int* out_coors_ptr, int* idx_map_ptr, int* count_ptr,
                        const int num_voxels, const int batch_size,
                        const int channels, const int max_voxels, const int H, const int W){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_voxels){
        return;
    }

    const int cur_batch = coors_ptr[idx * 4];
    const int cur_h = coors_ptr[idx * 4 + 2];
    const int cur_w = coors_ptr[idx * 4 + 3];

    int offset = cur_batch * H * W + cur_h * W + cur_w;
    bool valid = valid_mask_ptr[offset] > 0;
    if(valid){
        int old_num = atomicAdd(count_ptr + cur_batch, 1);
        for(int i=0; i < channels; ++i){
            out_features_ptr[(cur_batch * max_voxels + old_num) * channels + i] = features_ptr[idx * channels + i];
        }
        out_coors_ptr[(cur_batch * max_voxels + old_num) * 4] = coors_ptr[idx * 4];
        out_coors_ptr[(cur_batch * max_voxels + old_num) * 4 + 1] = coors_ptr[idx * 4 + 1];
        out_coors_ptr[(cur_batch * max_voxels + old_num) * 4 + 2] = coors_ptr[idx * 4 + 2];
        out_coors_ptr[(cur_batch * max_voxels + old_num) * 4 + 3] = coors_ptr[idx * 4 + 3];
        idx_map_ptr[cur_batch * max_voxels + old_num] = idx;
    }
}


__global__ void
foreground_gather_backward_kernel(const float* grad_ptr, const int* out_idx_map_ptr, float* out_grad_ptr,
                                  const int num_in_voxels, const int num_out_voxels, const int channels){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_out_voxels * channels){
        return;
    }

    const int cur_out_voxel = idx / channels;
    const int cur_channel = idx % channels;
    const int cur_in_voxel = out_idx_map_ptr[cur_out_voxel];

    out_grad_ptr[cur_in_voxel * channels + cur_channel] = grad_ptr[idx];
}

__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}


__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    // if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    int in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}

__global__ void
points_in_boxes_bev_kernel(const float* point_cloud_ptr, const float* boxes_ptr, int* out_mask_ptr,
                           const int num_points, const int num_point_attr, const int num_boxes, const int num_box_attr){
    const int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_points * num_boxes){
        return;
    }
    const int point_idx = idx % num_points;
    const int box_idx = idx / num_points;


    float local_x = 0, local_y = 0;
    int cur_in_flag = check_pt_in_box3d(point_cloud_ptr + point_idx * num_point_attr,
                                       boxes_ptr + box_idx * num_box_attr, local_x, local_y);
    out_mask_ptr[idx] = cur_in_flag;
}


void sparse_to_bev_sum_along_z_launcher(const float* features_ptr, const int* coors_ptr,
                                        float* bev_dense_ptr, int* valid_mask_ptr,
                                        const int num_voxels, const int batch_size,
                                        const int channels, std::vector<int>& spatial_shape){
    const int total_points = num_voxels * channels;
    int blockSize = THREADS_PER_BLOCK;
    int minGridSize = (total_points + blockSize - 1) / blockSize;
    sparse_to_bev_sum_along_z_kernel<<<minGridSize, blockSize, 0, 0>>>(features_ptr, coors_ptr, bev_dense_ptr, valid_mask_ptr,
                                                                        num_voxels, batch_size, channels,
                                                                        spatial_shape[1], spatial_shape[2], total_points);
}

void sparse_to_bev_sum_along_z_backward_launcher(const float* grad_ptr, const int* coors_ptr, float* output_grad_ptr,
                                                const int num_voxels, const int batch_size, const int channels,
                                                const int H, const int W){
    const int total_points = num_voxels * channels;
    int blockSize = THREADS_PER_BLOCK;
    int minGridSize = (total_points + blockSize - 1) / blockSize;
    sparse_to_bev_sum_along_z_backward_kernel<<<minGridSize, blockSize, 0, 0>>>(grad_ptr, coors_ptr, output_grad_ptr,
                                                                        num_voxels, batch_size, channels,
                                                                        H, W, total_points);
}


void foreground_gather_launcher(const float* features_ptr, const int* coors_ptr, const int* valid_mask_ptr,
                                float* out_features_ptr, int* out_coors_ptr, int* idx_map_ptr, int* count_ptr,
                                const int num_voxels, const int batch_size,
                                const int channels, const int max_voxels, std::vector<int>& spatial_shape){

    int blockSize = THREADS_PER_BLOCK;
    int minGridSize = (num_voxels + blockSize - 1) / blockSize;
    foreground_gather_kernel<<<minGridSize, blockSize, 0, 0>>>(features_ptr, coors_ptr, valid_mask_ptr,
                                                               out_features_ptr, out_coors_ptr, idx_map_ptr, count_ptr,
                                                                num_voxels, batch_size, channels, max_voxels,
                                                                spatial_shape[1], spatial_shape[2]);
}

void foreground_gather_backward_launcher(const float* grad_ptr, const int* out_idx_map_ptr, float* out_grad_ptr,
                                         const int num_in_voxels, const int num_out_voxels, const int channels){
    int blockSize = THREADS_PER_BLOCK;
    int minGridSize = (num_out_voxels * channels + blockSize - 1) / blockSize;
    foreground_gather_backward_kernel<<<minGridSize, blockSize, 0, 0>>>(grad_ptr, out_idx_map_ptr, out_grad_ptr,
                                                               num_in_voxels, num_out_voxels, channels);
}


void points_in_boxes_bev_launcher(const float* point_cloud_ptr, const float* boxes_ptr, int* out_mask_ptr,
                                const int num_points, const int num_point_attr, const int num_boxes, const int num_box_attr){

    int blockSize = THREADS_PER_BLOCK;
    int minGridSize = (num_points * num_boxes + blockSize - 1) / blockSize;
    points_in_boxes_bev_kernel<<<minGridSize, blockSize, 0, 0>>>(point_cloud_ptr, boxes_ptr, out_mask_ptr,
                                                                num_points, num_point_attr, num_boxes, num_box_attr);
}