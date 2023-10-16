import os
import torch
from torch.autograd import Function
import spconv
from .ops import SparseBEVToolsCUDA

class Sparse2BEVFunction(Function):
    @staticmethod
    def forward(ctx, features, coors, spatial_shape, batch_size):
        features = features.contiguous()
        coors = coors.contiguous()
        bev_dense, valid_mask = SparseBEVToolsCUDA.sparse_to_bev_sum_along_z(features, coors, spatial_shape, batch_size)
        ctx.save_for_backward(coors)
        return bev_dense, valid_mask

    @staticmethod
    def backward(ctx, grad1, grad2):
        grad1 = grad1.contiguous()
        coors,  = ctx.saved_tensors
        features_grad = SparseBEVToolsCUDA.sparse_to_bev_sum_along_z_backward(grad1, coors)
        return features_grad, None, None, None

class ForegroundGatherFunction(Function):
    @staticmethod
    def forward(ctx, features, coors, valid_mask, spatial_shape, batch_size):
        features = features.contiguous()
        coors = coors.contiguous()
        valid_mask = valid_mask.contiguous()
        max_voxels = 0
        for i in range(batch_size):
            max_voxels = max([max_voxels, (coors[:, 0]==i).sum().item()])
        voxel_features, voxel_coords, idx_map, count = SparseBEVToolsCUDA.foreground_gather(features, coors, valid_mask,
                                                                     spatial_shape, batch_size, max_voxels)
        out_voxels = []
        out_coords = []
        out_idx_map = []
        for i in range(batch_size):
            num = count[i].item()
            if num > 0:
                out_voxels.append(voxel_features[i][:num].contiguous().view(-1, features.size(1)))
                out_coords.append(voxel_coords[i][:num].contiguous().view(-1, 4))
                out_idx_map.append(idx_map[i][:num].contiguous().view(-1,))
        out_voxels = torch.cat(out_voxels, 0)
        out_coords = torch.cat(out_coords, 0)
        out_idx_map = torch.cat(out_idx_map, 0)

        ctx.save_for_backward(out_idx_map)
        ctx.num_in_voxels = features.size(0)
        return out_voxels, out_coords

    @staticmethod
    def backward(ctx, grad1, grad2):
        grad1 = grad1.contiguous()
        out_idx_map,  = ctx.saved_tensors
        num_in_voxels = ctx.num_in_voxels
        features_grad = SparseBEVToolsCUDA.foreground_gather_backward(grad1, out_idx_map, num_in_voxels)
        return features_grad, None, None, None, None

def sparse2bev(sparse_tensor: spconv.SparseConvTensor):
    features = sparse_tensor.features
    coors = sparse_tensor.indices
    spatial_shape = sparse_tensor.spatial_shape
    batch_size = sparse_tensor.batch_size
    bev_dense_tensor, bev_valid_mask = Sparse2BEVFunction.apply(features, coors, spatial_shape, batch_size)

    return bev_dense_tensor, bev_valid_mask


def foreground_gather(sparse_tensor: spconv.SparseConvTensor, valid_mask: torch.Tensor):
    features = sparse_tensor.features
    coors = sparse_tensor.indices
    spatial_shape = sparse_tensor.spatial_shape
    batch_size = sparse_tensor.batch_size
    valid_mask = valid_mask.int()
    voxel_features, voxel_coords = ForegroundGatherFunction.apply(features, coors, valid_mask,
                                                                  spatial_shape, batch_size)

    new_sparse_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=spatial_shape,
            batch_size=batch_size)

    return new_sparse_tensor


def foreground_gather_pytorch(sparse_tensor: spconv.SparseConvTensor, valid_mask: torch.Tensor):
    features = sparse_tensor.features
    coors = sparse_tensor.indices
    spatial_shape = sparse_tensor.spatial_shape
    batch_size = sparse_tensor.batch_size
    assert valid_mask.dim() == 3
    assert valid_mask.size(0) == batch_size

    valid_ = valid_mask[coors[:, 0].long(), coors[:, 2].long(), coors[:, 3].long()].to(torch.bool)

    voxel_features = features[valid_]
    voxel_coords = coors[valid_]

    new_sparse_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=spatial_shape,
            batch_size=batch_size)

    return new_sparse_tensor

def points_in_boxes_bev(point_cloud, boxes):
    """
    :param point_cloud: (N, C1), C1 >= 3
    :param boxes: (M, C2), C2 >= 7
    :return: (M, N)
    """
    output_mask = SparseBEVToolsCUDA.points_in_boxes_bev(point_cloud, boxes)
    return output_mask
