#include "libtorch_sdk.h"

std::vector<torch::Tensor>
sparse_to_bev_sum_along_z(torch::Tensor features, torch::Tensor coors, vector<int> spatial_shape, int batch_size){
    CHECK_INPUT(features);
    CHECK_INPUT(coors);
    auto features_type = features.scalar_type();
    auto coors_type = coors.scalar_type();
    const int channels = features.size(1);
    const int num_voxels = features.size(0);
    torch::Tensor bev_dense = torch::zeros({batch_size, channels, spatial_shape[1], spatial_shape[2]}, torch::dtype(features_type).device(features.device()));
    torch::Tensor valid_mask = torch::zeros({batch_size, spatial_shape[1], spatial_shape[2]}, torch::dtype(coors_type).device(features.device()));

    const float *features_ptr = features.data_ptr<float>();
    const int *coors_ptr = coors.data_ptr<int>();
    float *bev_dense_ptr = bev_dense.data_ptr<float>();
    int *valid_mask_ptr = valid_mask.data_ptr<int>();

    sparse_to_bev_sum_along_z_launcher(features_ptr, coors_ptr, bev_dense_ptr, valid_mask_ptr, num_voxels, batch_size, channels, spatial_shape);

    return {bev_dense, valid_mask};
}

torch::Tensor
sparse_to_bev_sum_along_z_backward(torch::Tensor grad, torch::Tensor coors){
    CHECK_INPUT(grad);
    CHECK_INPUT(coors);
    auto grad_type = grad.scalar_type();
    const int batch_size = grad.size(0);
    const int channels = grad.size(1);
    const int H = grad.size(2);
    const int W = grad.size(3);
    const int num_voxels = coors.size(0);
    torch::Tensor output_grad = torch::zeros({num_voxels, channels}, torch::dtype(grad_type).device(grad.device()));

    const float *grad_ptr = grad.data_ptr<float>();
    const int *coors_ptr = coors.data_ptr<int>();
    float *output_grad_ptr = output_grad.data_ptr<float>();

    sparse_to_bev_sum_along_z_backward_launcher(grad_ptr, coors_ptr, output_grad_ptr, num_voxels, batch_size, channels, H, W);

    return output_grad;
}


std::vector<torch::Tensor>
foreground_gather(torch::Tensor features, torch::Tensor coors, torch::Tensor valid_mask, std::vector<int> spatial_shape, int batch_size, int max_voxels){
    CHECK_INPUT(features);
    CHECK_INPUT(coors);
    CHECK_INPUT(valid_mask);
    auto features_type = features.scalar_type();
    auto coors_type = coors.scalar_type();
    const int channels = features.size(1);
    const int num_voxels = features.size(0);
    torch::Tensor out_features = torch::zeros({batch_size, max_voxels, channels}, torch::dtype(features_type).device(features.device()));
    torch::Tensor out_coors = torch::zeros({batch_size, max_voxels, 4}, torch::dtype(coors_type).device(features.device()));
    torch::Tensor idx_map = torch::zeros({batch_size, max_voxels}, torch::dtype(coors_type).device(features.device())) - 1;
    torch::Tensor count = torch::zeros({batch_size, }, torch::dtype(coors_type).device(features.device()));

    const float *features_ptr = features.data_ptr<float>();
    const int *coors_ptr = coors.data_ptr<int>();
    const int *valid_mask_ptr = valid_mask.data_ptr<int>();
    float *out_features_ptr = out_features.data_ptr<float>();
    int *out_coorsk_ptr = out_coors.data_ptr<int>();
    int *idx_map_ptr = idx_map.data_ptr<int>();
    int *count_ptr = count.data_ptr<int>();

    foreground_gather_launcher(features_ptr, coors_ptr, valid_mask_ptr, out_features_ptr, out_coorsk_ptr, idx_map_ptr, count_ptr,
                                num_voxels, batch_size, channels, max_voxels, spatial_shape);

    return {out_features, out_coors, idx_map, count};

}

torch::Tensor
foreground_gather_backward(torch::Tensor grad, torch::Tensor out_idx_map, int num_in_voxels){
    CHECK_INPUT(grad);
    CHECK_INPUT(out_idx_map);
    auto grad_type = grad.scalar_type();
    const int channels = grad.size(1);
    const int num_out_voxels = grad.size(0);
    torch::Tensor out_grad = torch::zeros({num_in_voxels, channels}, torch::dtype(grad_type).device(grad.device()));

    const float *grad_ptr = grad.data_ptr<float>();
    const int *out_idx_map_ptr = out_idx_map.data_ptr<int>();
    float *out_grad_ptr = out_grad.data_ptr<float>();

    foreground_gather_backward_launcher(grad_ptr, out_idx_map_ptr, out_grad_ptr, num_in_voxels, num_out_voxels, channels);

    return out_grad;

}


torch::Tensor
points_in_boxes_bev(torch::Tensor point_cloud, torch::Tensor boxes){
    CHECK_INPUT(point_cloud);
    CHECK_INPUT(boxes);

    const int num_points = point_cloud.size(0);
    const int num_point_attr = point_cloud.size(1);
    const int num_boxes = boxes.size(0);
    const int num_box_attr = boxes.size(1);

    torch::Tensor out_mask = torch::zeros({num_boxes, num_points}, torch::dtype(torch::kInt32).device(point_cloud.device()));

    const float *point_cloud_ptr = point_cloud.data_ptr<float>();
    const float *boxes_ptr = boxes.data_ptr<float>();
    int *out_mask_ptr = out_mask.data_ptr<int>();

    points_in_boxes_bev_launcher(point_cloud_ptr, boxes_ptr, out_mask_ptr, num_points, num_point_attr, num_boxes, num_box_attr);

    return out_mask;

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sparse_to_bev_sum_along_z", &sparse_to_bev_sum_along_z, "sparse_to_bev_sum_along_z");
  m.def("sparse_to_bev_sum_along_z_backward", &sparse_to_bev_sum_along_z_backward, "sparse_to_bev_sum_along_z_backward");
  m.def("foreground_gather", &foreground_gather, "foreground_gather");
  m.def("foreground_gather_backward", &foreground_gather_backward, "foreground_gather_backward");
  m.def("points_in_boxes_bev", &points_in_boxes_bev, "points_in_boxes_bev");

}
