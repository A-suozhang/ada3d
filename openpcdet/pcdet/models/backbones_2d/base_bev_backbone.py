import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd.function import Function

from .masked_bn import MaskedBatchNorm2d, MaskedBatchNorm2dV2, MaskedSyncBatchNormV2, sync_masked_batch_norm
from pcdet.models.backbones_3d.drop_utils import check_inbox, get_topk_mask

class BaseBEVBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg
        self.cfg = None # filled in predictor_forward

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        if self.model_cfg.get('MASKED_BATCHNORM', None) is not None:
            if self.model_cfg.get('MASKED_BATCHNORM', None):
                if self.model_cfg.get('MASKED_BATCHNORM', None) == 'v2':
                    self.masked_batchnorm = True
                    bn_type = MaskedBatchNorm2dV2
                else:
                    self.masked_batchnorm = True
                    bn_type = MaskedBatchNorm2d
            else:
                self.masked_batchnorm = False
                bn_type = nn.BatchNorm2d
        else:
            self.masked_batchnorm = False
            bn_type = nn.BatchNorm2d

        if self.model_cfg.get('BN_AFFINE',None) is not None:
            if self.model_cfg.get('BN_AFFINE',None):
                self.bn_affine=True
            else:
                self.bn_affine=False
        else:
            self.bn_affine=True   # INFO: when not having this cfg, default is True

        self.save_dict = {}

        # CFG: probe the feature_maps
        PROBE_FEATURE = False
        self.probe_feature = PROBE_FEATURE

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                bn_type(num_filters[idx], eps=1e-3, momentum=0.01, affine=self.bn_affine),  # DEBUG_ONLY: affine=false
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    bn_type(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        bn_type(num_upsample_filters[idx], eps=1e-3, momentum=0.01, affine=self.bn_affine),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        bn_type(num_upsample_filters[idx], eps=1e-3, momentum=0.01,affine=self.bn_affine),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                bn_type(c_in, eps=1e-3, momentum=0.01,affine=self.bn_affine),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict, enable_predictor=False, predictor_args=None):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """

        if enable_predictor:
            # unwrap predictor
            predictor = predictor_args['predictor']
            predictor_location_2d = predictor_args['location_2d']
            n_predictor_location_2d_each_block = len(predictor_location_2d)
            n_predictor_location_3d = len(predictor_args['location_3d'])
            predictor_preprocess = predictor_args['predictor_preprocess'][n_predictor_location_3d:]
            hard_drop = predictor_args['hard_drop']
            drop_scheme = predictor_args['drop_scheme']
            if_save_runtime = predictor_args['if_save_runtime']

            pred_heatmap = {}

        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        h_shape, w_shape = spatial_features.shape[2], spatial_features.shape[3]
        inbox_rate_d = {}
        for i_block in range(len(self.blocks)):
            n_op_per_block = len(self.blocks[i_block])
            # x = self.blocks[i](x)
            if enable_predictor:
                conv_idx = []
                for i_, m_ in enumerate(self.blocks[i_block].modules()): # split the module exec for save activation\
                    if isinstance(m_,nn.Conv2d):
                        conv_idx.append(i_)
                predictor_location_2d_idx = [conv_idx[predictor_location_2d_] for predictor_location_2d_ in predictor_location_2d[i_block]]

            for i_, m_ in enumerate(self.blocks[i_block].modules()):
                if i_ > 0:  # 1st module is the whole nn.Sequential() skip em
                    i_with_block = i_ + i_block*n_op_per_block

                    # INFO: apply predictor inference before module inference
                    # DEBUG: the base_bev_backbone has a zero_padding, making the 176 -> 178
                    if enable_predictor:
                        if i_ in predictor_location_2d_idx:
                            location_idx_ = predictor_location_2d_idx.index(i_) + n_predictor_location_2d_each_block*i_block
                            x_pooled = predictor_preprocess[location_idx_][0](x.detach())    # the predictor loss grad should not have effects on backbone weights
                            x_pooled = predictor_preprocess[location_idx_][1](x_pooled)
                            x_pooled = x_pooled[:,:,:h_shape,:w_shape] # get the upper-left for predictor_input in case differnet resolution
                            bev_valid_mask = (x_pooled.sum(1,keepdim=True) != 0).float()
                            upsample_func = predictor_preprocess[location_idx_][2]
                            if isinstance(self.k_percent, list):
                                assert len(self.k_percent) == len(predictor_location_2d_idx)*n_predictor_location_2d_each_block
                                k_percent_ = self.k_percent[location_idx_]
                            else:
                                assert isinstance(self.k_percent, int)
                                k_percent_ = self.k_percent


                            # INFO: apply hard drop for 2D backbone
                            pred_heatmap_ = predictor(x_pooled)
                            pred_heatmap['2d_{}'.format(conv_idx.index(i_)+i_block*n_op_per_block)] = pred_heatmap_
                            pred_heatmap_ = pred_heatmap_.mean(dim=1, keepdim=True)
                            pred_heatmap_ = upsample_func(pred_heatmap_)[:,:,:h_shape,:w_shape]
                            bev_valid_mask = upsample_func(bev_valid_mask)
                            # debug_only
                            # sparse_rate = lambda x: (torch.sum(x == 0) / x.numel())
                            # if sparse_rate(pred_heatmap_) > 0.9:
                                # import ipdb; ipdb.set_trace()

                            if drop_scheme == 'plain':
                                importance_score_ = pred_heatmap_
                                # pred_heatmap_valid_mask = (importance_score_ > 0.45*importance_score_.max()).float()  # [N,C,H,W]
                                # pred_heatmap_valid_mask = (importance_score_ > importance_score_.median()).float()  # [N,C,H,W]
                                # pred_heatmap_valid_mask = get_topk_mask(importance_score_, k_percent=k_percent) # [N,C,H,W]
                                pred_heatmap_valid_mask, sparse_rate_pre, sparse_rate_post= get_topk_mask(importance_score_*bev_valid_mask,k_percent_)
                            elif drop_scheme == 'density':
                                import ipdb; ipdb.set_trace()
                                kernel_size = 8
                                alpha = 0.15
                                eps = 1.e-4
                                output_size = [pred_heatmap_.shape[2],pred_heatmap_.shape[3]]
                                density_heatmap_ = (bev_valid_mask != 0).float()
                                density_heatmap_ = F.avg_pool2d(density_heatmap_, kernel_size=kernel_size, stride=kernel_size)
                                density_heatmap_ = F.interpolate(density_heatmap_,size=output_size, mode='nearest')
                                importance_score_ = pred_heatmap_/(torch.pow(density_heatmap_+eps, alpha))
                                pred_heatmap_valid_mask, sparse_rate_pre, sparse_rate_post= get_topk_mask(importance_score_,k_percent_)

                                # debug_only
                                # k_percent = 50

                                # pred_heatmap_valid_mask_plain, sparse_rate_pre, sparse_rate_post= get_topk_mask(pred_heatmap_,k_percent)
                                # print('plain drop',sparse_rate_pre,sparse_rate_post)
                                # pred_heatmap_valid_mask, sparse_rate_pre, sparse_rate_post= get_topk_mask(importance_score_,k_percent)
                                # print('density drop',sparse_rate_pre,sparse_rate_post)

                                # save_d = {}
                                # save_d['pred_heatmap'] = pred_heatmap_.squeeze(1)  # [4,1,W,H]
                                # save_d['density_heatmap'] = density_heatmap_.float().squeeze(1)
                                # save_d['importance_score_'] = importance_score_.squeeze(1)
                                # inbox_rate, check_inbox_save_d = check_inbox(self.cfg, data_dict, (pred_heatmap_*bev_valid_mask).squeeze(1), (pred_heatmap_valid_mask*bev_valid_mask).squeeze(1), if_save_runtime=True)
                                # inbox_rate_plain, _ = check_inbox(self.cfg, data_dict, (pred_heatmap_*bev_valid_mask).squeeze(1), (pred_heatmap_valid_mask_plain*bev_valid_mask).squeeze(1), if_save_runtime=False)
                                # save_d['dropped'] = check_inbox_save_d['dropped'].float().permute([0,2,1])
                                # save_d['boxes'] = (check_inbox_save_d['masks']==0).float().permute([0,2,1])
                                # for k,v in save_d.items():
                                    # print(k, v.shape)
                                # save_d['inbox_rate'] = inbox_rate
                                # save_d['inbox_rate_plain'] = inbox_rate_plain
                                # print('inbox_rate_plain: {},  inbox_rate: {}'.format(inbox_rate_plain, inbox_rate))

                                # torch.save(save_d,'./debug_density.pth')
                                # import ipdb; ipdb.set_trace()

                            else:
                                raise NotImplementedError

                            # INFO: check the inbox rate
                            upsampled_pred_heatmap_valid_mask = pred_heatmap_valid_mask  # alreadly upsample the pred_heatmap, so produced valid_mask no need for upsample
                            inbox_rate, check_inbox_save_d = check_inbox(self.cfg, data_dict, (pred_heatmap_*bev_valid_mask).squeeze(1), (pred_heatmap_valid_mask*bev_valid_mask).squeeze(1), if_save_runtime=if_save_runtime)
                            self.save_dict['boxes_2d_{}'.format(i_with_block)] = check_inbox_save_d
                            inbox_rate_d['pred_2d_{}_inbox_rate'.format(i_with_block)] = inbox_rate

                            if hard_drop:
                                # print('drop sparse rate',upsampled_pred_heatmap_valid_mask.sum() / upsampled_pred_heatmap_valid_mask.nelement())
                                # print('drop before 2d feature', (x!=0).sum() / x.nelement())
                                with torch.autograd.set_detect_anomaly(True):
                                    enlarged_mask = torch.zeros_like(x,device=x.device)
                                    enlarged_mask[:,:,:h_shape,:w_shape] = upsampled_pred_heatmap_valid_mask
                                    # x[:,:,:h_shape,:w_shape] = StraightThroughSparseMask.apply(x[:,:,:h_shape,:w_shape], upsampled_pred_heatmap_valid_mask)   # apply sparse mask
                                    x_undropped = x
                                    x = StraightThroughSparseMask.apply(x, enlarged_mask)   # apply sparse mask
                                    n_pre_drop = x_undropped.sum(1).nonzero().shape[0]
                                    n_post_drop = x.sum(1).nonzero().shape[0]
                                    # print(n_pre_drop, n_post_drop)
                                    inbox_rate_d['pred_2d_{}_drop_rate'.format(i_with_block)] = float((n_pre_drop - n_post_drop)/n_pre_drop)
                                    # inbox_rate_d['pred_2d_{}_drop_rate'.format(i_)] = float(((upsampled_pred_heatmap_valid_mask==0).int().sum() / upsampled_pred_heatmap_valid_mask.nelement()).cpu())

                    x = m_(x)

                    if self.probe_feature:
                        self.save_dict['block{}_{}'.format(i_block,i_)] = x.max(1)[0].detach().data  # get max along C dim
                        # print('block{}_{}'.format(i,i_),x.shape, m_)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x

            if len(self.deblocks) > 0:
                # ups.append(self.deblocks[i](x))
                x_ = x
                for i_, m_ in enumerate(self.deblocks[i_block].modules()): # split the module exec for save activation
                    if i_ > 0:
                        x_ = m_(x_)
                        if self.probe_feature:
                            self.save_dict['deblock{}_{}'.format(i_block,i_)] = x.max(1)[0].detach().data  # get max along C dim
                ups.append(x_)
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        if enable_predictor:
            data_dict.update(inbox_rate_d)
            return data_dict, pred_heatmap
        else:
            return data_dict


class BaseBEVBackboneV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        layer_nums = self.model_cfg.LAYER_NUMS
        num_filters = self.model_cfg.NUM_FILTERS
        assert len(layer_nums) == len(num_filters) == 2

        num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
        upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        assert len(num_upsample_filters) == len(upsample_strides)

        num_levels = len(layer_nums)
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    num_filters[idx], num_filters[idx], kernel_size=3,
                    stride=1, padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride >= 1:
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['multi_scale_2d_features']

        x_conv4 = spatial_features['x_conv4']
        x_conv5 = spatial_features['x_conv5']

        ups = [self.deblocks[0](x_conv4)]

        x = self.blocks[1](x_conv5)
        ups.append(self.deblocks[1](x))

        x = torch.cat(ups, dim=1)
        x = self.blocks[0](x)

        data_dict['spatial_features_2d'] = x

        return data_dict

class StraightThroughSparseMask(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, sparse_mask):
        ctx.save_for_backward(sparse_mask)
        return input*sparse_mask

    @staticmethod
    def backward(ctx, grad_output):
        sparse_mask = ctx.saved_tensors[0]  # returns a tuple
        return grad_output*sparse_mask, None

