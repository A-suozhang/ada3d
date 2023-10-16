import torch.nn as nn
import torch


class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        # self.bev_density_map = self.get_density_map(encoded_spconv_tensor)
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict
    
    def get_density_map(self, spconv_tensor):
        pass

        # The Accurate way: close to simple pooling
        # spatial_shape = spconv_tensor.spatial_shape
        # voxel_coords = spconv_tensor.indices
        # batch_size = spconv_tensor.batch_size
        # pool_size = 4
        # for i_bs in range(batch_size):
            # voxels_cur_batch = voxel_coords[torch.where(voxel_coords[:,0]==i_bs)[0],1:]   # [N,3]
            # flattened_voxel_coords = voxel_coords[:,1]//pool_size + (voxel_coords[:,2]//pool_size)*(spatial_shape[1]//pool_size)
            # unique_voxel_coords, unique_counts = flattened_voxel_coords.unique(return_counts=True)
            # unique_voxel_coords = torch.stack([unique_voxel_coords//(spatial_shape[1]//pool_size),unique_voxel_coords%(spatial_shape[1]//pool_size)],dim=-1)

