
# SparseBEVTools


## Install
```bash
python setup.py develop
pip list
```

## Sample
```python
from SparseBEVTools import sparse2bev, foreground_gather, foreground_gather_pytorch
import torch
import spconv

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
```
