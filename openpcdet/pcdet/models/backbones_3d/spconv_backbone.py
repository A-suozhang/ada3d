from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ...utils.spconv_utils import replace_feature, spconv
from SparseBEVTools import sparse2bev, foreground_gather, foreground_gather_pytorch
from pcdet.models.backbones_3d.drop_utils import check_inbox, get_topk_mask


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.cfg = None  # filled in predictor_forward
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

        self.save_dict = {}
        PROBE_FEATURE = False
        if PROBE_FEATURE:
            self.probe_feature = True
        else:
            self.probe_feature = False

    def forward(self, batch_dict, enable_predictor=False, predictor_args=None):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        if enable_predictor:
             # unwrap predictor args
            predictor = predictor_args['predictor']
            predictor_location_3d = predictor_args['location_3d']
            predictor_preprocess = predictor_args['predictor_preprocess'][:len(predictor_location_3d)]
            hard_drop = predictor_args['hard_drop']
            drop_scheme = predictor_args['drop_scheme']
            if_save_runtime = predictor_args['if_save_runtime']
            pred_heatmap = {}

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        if self.probe_feature:
            self.save_dict['3d_conv_input_pre'] = input_sp_tensor

        # x_conv1 = self.conv1(x)
        # x_conv2 = self.conv2(x_conv1)
        # x_conv3 = self.conv3(x_conv2)
        # x_conv4 = self.conv4(x_conv3)

        conv_list = [self.conv1, self.conv2, self.conv3, self.conv4]
        act_list = [x]
        inbox_rate_d = {}
        for idx_, conv_ in enumerate(conv_list):
            conv_out = conv_(act_list[idx_])
            conv_out_undropped = conv_out
            # act_list.append(conv_(act_list[idx_]))

            if enable_predictor: # apply predictor after conv module
                if idx_ in predictor_location_3d:
                    location_idx_ = predictor_location_3d.index(idx_)
                    input_sp_tensor_pooled = predictor_preprocess[location_idx_][0](conv_out)    # apply the MaxPool3d
                    bev_dense_tensor, bev_valid_mask = sparse2bev(input_sp_tensor_pooled)
                    bev_dense_tensor = predictor_preprocess[location_idx_][1](bev_dense_tensor.detach()) # detach the tensor, predictor loss should not backward into backbone weights
                    pred_heatmap_ = predictor(bev_dense_tensor*bev_valid_mask.unsqueeze(1))
                    pred_heatmap['3d_{}'.format(idx_)] = pred_heatmap_
                    bev_valid_mask = bev_valid_mask.unsqueeze(1)

                    # INFO: the evaluation metric for dropping voxel
                    # - default: use the pred_heatmap's sum
                    h_shape, w_shape = conv_out_undropped.spatial_shape[1], conv_out_undropped.spatial_shape[2]
                    pred_heatmap_ = pred_heatmap_.mean(dim=1, keepdim=True)  # [BS,W,H]
                    # print(pred_heatmap_.max(), pred_heatmap_.min())  # within range ~(0,1)
                    # print((pred_heatmap_>0.6*pred_heatmap_.max()).int().sum()/pred_heatmap_.nelement(sum))
                    if isinstance(self.k_percent, list):
                        assert len(self.k_percent) == len(predictor_location_3d)
                        k_percent_ = self.k_percent[location_idx_]
                    else:
                        assert isinstance(self.k_percent, int)
                        k_percent_ = self.k_percent

                    if drop_scheme == 'plain':
                        importance_score_ = pred_heatmap_
                        # INFO: simply drop topk, not so slow
                        pred_heatmap_valid_mask, sparse_rate_pre, sparse_rate_post= get_topk_mask(importance_score_*bev_valid_mask,k_percent_)   # only drop the un-sparse location
                        # pred_heatmap_valid_mask = (importance_score_ > importance_score_.median()).float()  # [N,C,H,W]
                        # pred_heatmap_valid_mask = (importance_score_ > 0.5*importance_score_.max()).float()  # [N,C,H,W]

                    elif drop_scheme == 'density':
                        import ipdb; ipdb.set_trace()
                        kernel_size = 16
                        alpha = 0.2
                        eps = 1.e-4
                        output_size = [pred_heatmap_.shape[2],pred_heatmap_.shape[3]]
                        density_heatmap_ = (bev_valid_mask != 0).float()
                        density_heatmap_ = F.avg_pool2d(density_heatmap_, kernel_size=kernel_size, stride=kernel_size)
                        density_heatmap_ = F.interpolate(density_heatmap_,size=output_size, mode='nearest')
                        importance_score_ = pred_heatmap_/(torch.pow(density_heatmap_+eps, alpha))
                        pred_heatmap_valid_mask, sparse_rate_pre, sparse_rate_post= get_topk_mask(importance_score_,k_percent)

                        # -------------------------- debug_only ----------------------------
                        # k_percent = 50

                        # pred_heatmap_valid_mask_plain, sparse_rate_pre, sparse_rate_post= get_topk_mask(pred_heatmap_,k_percent)
                        # print('plain drop',sparse_rate_pre,sparse_rate_post)
                        # pred_heatmap_valid_mask, sparse_rate_pre, sparse_rate_post= get_topk_mask(importance_score_,k_percent)
                        # print('density drop',sparse_rate_pre,sparse_rate_post)

                        # save_d = {}
                        # save_d['pred_heatmap'] = pred_heatmap_.squeeze(1)  # [4,1,W,H]
                        # save_d['density_heatmap'] = density_heatmap_.squeeze(1)
                        # save_d['importance_score_'] = importance_score_.squeeze(1)
                        # inbox_rate, check_inbox_save_d = check_inbox(self.cfg, batch_dict, (pred_heatmap_*bev_valid_mask).squeeze(1), (pred_heatmap_valid_mask*bev_valid_mask).squeeze(1), if_save_runtime=True)
                        # inbox_rate_plain, _ = check_inbox(self.cfg, batch_dict, (pred_heatmap_*bev_valid_mask).squeeze(1), (pred_heatmap_valid_mask_plain*bev_valid_mask).squeeze(1), if_save_runtime=False)
                        # save_d['dropped'] = check_inbox_save_d['dropped'].float().permute([0,2,1])
                        # save_d['boxes'] = (check_inbox_save_d['masks']==0).float().permute([0,2,1])
                        # for k,v in save_d.items():
                            # print(k, v.shape)
                        # save_d['inbox_rate'] = inbox_rate
                        # save_d['inbox_rate_plain'] = inbox_rate_plain
                        # print('inbox_rate_plain: {},  inbox_rate: {}'.format(inbox_rate_plain, inbox_rate))

                        # torch.save(save_d,'./debug_density.pth')
                        # import ipdb; ipdb.set_trace()
                        # pred_heatmap_valid_mask = ((importance_score_ > 0.5)*importance_score_.max()).float()
                    else:
                        raise NotImplementedError

                    upsample_func = predictor_preprocess[location_idx_][2]
                    # INFO: get the inbox rate in BEV space
                    inbox_rate, check_inbox_save_d = check_inbox(self.cfg, batch_dict, (pred_heatmap_*bev_valid_mask).squeeze(1), (pred_heatmap_valid_mask*bev_valid_mask).squeeze(1), if_save_runtime=if_save_runtime)
                    self.save_dict['boxes_3d_{}'.format(idx_)] = check_inbox_save_d
                    inbox_rate_d['pred_3d_{}_inbox_rate'.format(idx_)] = float(inbox_rate)

                    upsampled_pred_heatmap_valid_mask = upsample_func(pred_heatmap_valid_mask)[:,:,:h_shape,:w_shape].squeeze(1)
                    upsampled_pred_heatmap_ = upsample_func(pred_heatmap_)[:,:,:h_shape,:w_shape].squeeze(1)
                    # print(upsampled_pred_heatmap_valid_mask.sum() / upsampled_pred_heatmap_valid_mask.numel())

                    # if enable hard_drop, actually drop the activation and reinit the sparse tensor
                    if hard_drop:
                        conv_out = foreground_gather_pytorch(conv_out, upsampled_pred_heatmap_valid_mask)
                        inbox_rate_d['pred_3d_{}_drop_rate'.format(idx_)] = float(1 - (conv_out.features.shape[0] / conv_out_undropped.features.shape[0]))
                        # print(conv_out_undropped.features.shape, conv_out.features.shape)

            if self.probe_feature:
                self.save_dict['3d_conv_{}_pre_drop'.format(idx_+1)] = conv_out_undropped
                self.save_dict['3d_conv_{}_post_drop'.format(idx_+1)] = conv_out

            act_list.append(conv_out)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        if self.probe_feature:
            self.save_dict['3d_conv_out_pre'] = act_list[-1]
        out = self.conv_out(act_list[-1])

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': act_list[1],
                'x_conv2': act_list[2],
                'x_conv3': act_list[3],
                'x_conv4': act_list[4],
            }
        })

        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        if enable_predictor:
            batch_dict.update(inbox_rate_d)
            return batch_dict, pred_heatmap
        else:
            return batch_dict


