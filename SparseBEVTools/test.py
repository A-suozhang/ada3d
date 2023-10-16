from SparseBEVTools import sparse2bev, foreground_gather, foreground_gather_pytorch
import torch
import spconv
import numpy as np


def save_pcd(data, path):

    data = data[:, :4]
    # data = data[data[:, 1] > 10]
    if data.shape[1] == 3:
        data = np.pad(data, ((0, 0), (0, 1)), mode='constant', constant_values=1.0)
    elif data.shape[1] < 3:
        raise RuntimeError('error input shape.')

    header = ["# .PCD v0.7 - Point Cloud Data file format",
              "VERSION 0.7",
              "FIELDS x y z intensity",
              "SIZE 4 4 4 4",
              "TYPE F F F F",
              "COUNT 1 1 1 1",
              "WIDTH %d" % len(data),
              "HEIGHT 1",
              "VIEWPOINT 0 0 0 1 0 0 0",
              "POINTS %d" % len(data),
              "DATA ascii"]

    with open(path, 'w') as f:
        for line in header:
            f.write(line + '\n')
        for item in data:
            x, y, z, inten = item
            line = "%.5f %.5f %.5f %.5f" % (x, y, z, inten)
            f.write(line + '\n')




batch_size = 2
num_voxels = 10000
num_features = 16
sparse_shape = [40, 1400, 1600]

voxel_features = torch.randn(size=(num_voxels, num_features))
voxel_coords_x = torch.randint(low=0, high=sparse_shape[0] - 1, size=(num_voxels, )).int()
voxel_coords_y = torch.randint(low=0, high=sparse_shape[1] - 1, size=(num_voxels, )).int()
voxel_coords_z = torch.randint(low=0, high=sparse_shape[2] - 1, size=(num_voxels, )).int()
voxel_coords_bs = torch.zeros_like(voxel_coords_x, dtype=torch.int32)
voxel_coords = torch.stack([voxel_coords_bs, voxel_coords_x, voxel_coords_y, voxel_coords_z]).t().contiguous()
voxel_coords[num_voxels//3:, 0] = 1
voxel_coords = voxel_coords.int()

voxel_features = voxel_features.cuda()
voxel_coords = voxel_coords.cuda()
voxel_features.requires_grad = True

spconv_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords,
            spatial_shape=sparse_shape,
            batch_size=batch_size
        )

bev_dense_tensor, bev_valid_mask = sparse2bev(spconv_tensor)
(bev_dense_tensor**2).sum().backward()
print(voxel_features)
print(voxel_features.grad)


# 更新valid mask
rand = torch.randn_like(bev_valid_mask, dtype=torch.float32)
bev_valid_mask = bev_valid_mask & (rand > 0)

voxel_features.grad.fill_(0.0)
new_spconv_tensor = foreground_gather(spconv_tensor, bev_valid_mask)
(new_spconv_tensor.features**2).sum().backward()
grad1 = voxel_features.grad
print(voxel_features.grad)
print(new_spconv_tensor.features.size(0))


print("***************************************************")
voxel_features.grad.fill_(0.0)
new_spconv_tensor1 = foreground_gather_pytorch(spconv_tensor, bev_valid_mask)
(new_spconv_tensor1.features**2).sum().backward()
grad2 = voxel_features.grad
print(voxel_features.grad)
print(new_spconv_tensor1.features.size(0))

print('error:', (grad1 - grad2).abs().sum())



def test_points_in_boxes_bev():
    from SparseBEVTools import points_in_boxes_bev

    x = np.linspace(0.0, 100, 1000)
    y = np.linspace(-50, 50, 1000)
    points = np.meshgrid(x, y)
    points = np.stack([*points, np.zeros_like(points[0]), np.ones_like(points[0])], -1)
    points = points.reshape(-1, 4)
    points = torch.from_numpy(points).cuda().float()
    print(points)

    boxes = torch.Tensor([[25, -6, 8.0, 6, 3.5, 0.8, 0.5],
                          [25, 16.6, -8.0, 6, 3.5, 0.8, -0.5]]).float().cuda()

    mask = points_in_boxes_bev(points, boxes)
    save_pcd(points[mask[0] > 0].view(-1, 4).cpu().numpy(), 'point_cloud_1.pcd')
    save_pcd(points[mask[1] > 0].view(-1, 4).cpu().numpy(), 'point_cloud_2.pcd')

    print(mask.sum())

if __name__ == '__main__':
    test_points_in_boxes_bev()
