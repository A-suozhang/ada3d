from .detector3d_template import Detector3DTemplate
from SparseBEVTools import sparse2bev, foreground_gather, foreground_gather_pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import spconv

import pcdet.utils.loss_utils as loss_utils
from pcdet.models.backbones_2d.masked_bn import MaskedBatchNorm2d, MaskedBatchNorm2dV2


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()
        # INFO: define the model save_dict and define tensors_to_save
        keys_to_save = []
        self.save_dict = {}
        for k_ in keys_to_save:
            self.save_dict[k_] = []

        # INFO: flag to denote whether add to save_dict, changes in train_utils.py
        self.if_save_runtime = False

        if hasattr(model_cfg,'PREDICTOR'):
            self.predictor_stride = model_cfg.PREDICTOR.FEATURE_MAP_STRIDE
            self.predictor_loss_lambda = model_cfg.PREDICTOR.LOSS_LAMBDA
            # self.grid_size # defined in train_predictor_one_epoch
            layer_nums = model_cfg.PREDICTOR.LAYER_NUMS
            layer_strides = model_cfg.PREDICTOR.LAYER_STRIDES
            num_filters = model_cfg.PREDICTOR.NUM_FILTERS
            num_groups = model_cfg.PREDICTOR.NUM_GROUPS
            num_expansions = model_cfg.PREDICTOR.NUM_EXPANSIONS
            expansion_in = [num_expansions[0]] + num_expansions
            expansion_out = num_expansions + [num_expansions[-1]]
            if self.model_cfg.PREDICTOR.get('MASKED_BATCHNORM', None):
                if self.model_cfg.PREDICTOR.get('MASKED_BATCHNORM', None) == 'v2':
                    self.masked_batchnorm = True
                    bn_type = MaskedBatchNorm2dV2
                else:
                    self.masked_batchnorm = True
                    bn_type = MaskedBatchNorm2d
            else:
                self.masked_batchnorm = False
                bn_type = nn.BatchNorm2d

            for idx_ in range(len(layer_nums)): # DEBUG: donot support multi-levels predictor
                cur_layers = []
                for i_ in range(layer_nums[idx_]):
                    cur_layers.extend([
                        nn.Conv2d(num_filters[idx_]*expansion_in[i_] ,num_filters[idx_]*expansion_out[i_],kernel_size=3,padding=1,bias=False,groups=num_groups[idx_]),
                        bn_type(num_filters[idx_]*expansion_out[i_],eps=1e-3,momentum=0.01),
                        nn.Sigmoid()
                        ])
            cur_layers.extend([
                nn.Conv2d(num_filters[-1]*expansion_out[-1],self.num_class,kernel_size=1,bias=False),
                bn_type(self.num_class,eps=1e-3,momentum=0.01),
                nn.Sigmoid()
                ])
            self.predictor = nn.Sequential(*cur_layers)


            # TODO: get the location_at 3d/2d backbone's feature's preprocessor!
            # DEBUG: noted that the stride may not generalize, may need extra check when changing backbone* 
            self.predictor_preprocess = nn.ModuleList()
            self.location_3d = model_cfg.PREDICTOR.LOCATION_3D
            # dirty fix: in some old cfgs, since there is only one block, the location are like [1,3] if so, make it [[1,3]] to adapt new version
            if not isinstance(model_cfg.PREDICTOR.LOCATION_2D[0],list):
                model_cfg.PREDICTOR.LOCATION_2D = [model_cfg.PREDICTOR.LOCATION_2D]
            self.location_2d = model_cfg.PREDICTOR.LOCATION_2D
            for localtion_3d_ in model_cfg.PREDICTOR.LOCATION_3D:
                in_channel_3d = self.module_list[1].backbone_channels['x_conv{}'.format(localtion_3d_+1)]
                # in_channel_3d = getattr(self.module_list[1],'conv{}'.format(localtion_3d_+1))[0].conv1.in_channels
                stride_3d = 1 if localtion_3d_ == 0 else 2*localtion_3d_
                pool_3d_kernel_size = self.predictor_stride // stride_3d
                assert pool_3d_kernel_size >= 1
                # initiialize the head before feed into predictor
                cur_layer = nn.ModuleList([
                        spconv.SparseMaxPool3d(kernel_size=pool_3d_kernel_size) if pool_3d_kernel_size>1 else nn.Identity(),
                        nn.Conv2d(in_channel_3d, num_filters[0]*expansion_in[0], kernel_size=1,bias=False),
                        nn.Upsample(scale_factor=pool_3d_kernel_size,mode='nearest')
                        ])
                self.predictor_preprocess.append(cur_layer)

            # define how many voxels to drop
            self.module_list[1].k_percent = model_cfg.PREDICTOR.K_PERCENT_3D
            self.module_list[3].k_percent = model_cfg.PREDICTOR.K_PERCENT_2D

            stride_ = 1
            for i_block, block_ in enumerate(self.module_list[3].blocks):
                accumulated_strides = []
                for m in self.module_list[3].blocks[i_block].modules():
                    if isinstance(m,nn.Conv2d):
                        accumulated_strides.append(stride_*m.stride[0])
                        stride_ *= m.stride[0]
                for location_2d_ in model_cfg.PREDICTOR.LOCATION_2D[i_block]:
                    # DEBUG: for now, assume that the base_bev_backbone has only one module, and the predictor is only applied on the 1st module
                    # INFO: make the preprocess for predictor reading the cfg location_2d
                    in_channel_2d = self.module_list[3].blocks[i_block][1+3*location_2d_].in_channels
                    stride_2d = accumulated_strides[location_2d_]
                    # INFO: for centerpoint, the backbone3d feature map is 8
                    # bigger stride, smaller w,h, need to upsample first
                    pool_2d_kernel_size = 8*stride_2d / self.predictor_stride
                    if pool_2d_kernel_size == 1:
                        preprocess_op = nn.Identity()
                        postprocess_op = nn.Identity()
                    elif pool_2d_kernel_size > 1:
                        assert pool_2d_kernel_size % 1 == 0
                        pool_2d_kernel_size = int(pool_2d_kernel_size)
                        preprocess_op = nn.Upsample(scale_factor=pool_2d_kernel_size)
                        postprocess_op = nn.MaxPool2d(kernel_size=pool_2d_kernel_size)
                    else:
                        upsample_kernel_size = 1/pool_2d_kernel_size
                        assert upsample_kernel_size % 1 == 0
                        upsample_kernel_size = int(upsample_kernel_size)
                        preprocess_op = nn.MaxPool2d(kernel_size=upsample_kernel_size)
                        postprocess_op = nn.Upsample(scale_factor=upsample_kernel_size, mode='nearest')

                    cur_layer = nn.ModuleList([
                        preprocess_op,
                        nn.Conv2d(in_channel_2d,num_filters[0]*expansion_in[0], kernel_size=1,bias=False),
                        postprocess_op
                        ])
                    self.predictor_preprocess.append(cur_layer)
        else:
            pass

    def forward(self, batch_dict):

        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

         # INFO: save some tensors for debugging
        if self.if_save_runtime:
            # merge the submodules save_dict into model_ save_dict
            for cur_module in self.module_list:
                if hasattr(cur_module,"save_dict"):
                    for k_,v_ in cur_module.save_dict.items():
                        if k_ not in self.save_dict:
                            self.save_dict[k_] = []
                        self.save_dict[k_].append(v_)
            # save some vars directly into model,save_dict
            # self.save_dict['points'].append(batch_dict['points'].detach().data)


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

    def predictor_forward(self, batch_dict, hard_drop=False, get_normal_loss=False):
        # cur_module: 
        #   0 - VFE
        #   1 - backbone_3d
        #   2 - height_compression
        #   3 - backbone_2d
        #   4 - heads

        pred_heatmap_d = {}
        for idx_, cur_module in enumerate(self.module_list):

            if idx_ == 1 or idx_ == 3: # for backbone_3d/2d
                # 1. get the predicted heatmap from predictor
                if self.model_cfg.PREDICTOR.get('DROP_SCHEME') is not None:
                    drop_scheme = self.model_cfg.PREDICTOR.DROP_SCHEME
                else:
                    drop_scheme = 'plain'

                predictor_args = {
                        'predictor': self.predictor,   # predictor is a module_list, [0] is nn.sequential
                        'predictor_preprocess': self.predictor_preprocess,
                        'location_3d': self.location_3d,
                        'location_2d': self.location_2d,
                        'hard_drop': hard_drop,    # whether actually drop the input
                        'drop_scheme': drop_scheme,
                        'if_save_runtime': self.if_save_runtime,
                        }

                batch_dict, pred_heatmap = cur_module(batch_dict, enable_predictor=True, predictor_args=predictor_args)

                pred_heatmap_to_save = {}
                for k,v in pred_heatmap.items():
                    pred_heatmap_to_save[k] = v.detach().cpu().data
                pred_heatmap_d.update(pred_heatmap)
            else:
                batch_dict = cur_module(batch_dict)

        if not get_normal_loss:
            # INFO: get gt_heatmap
            # DEBUG: when put this before model inference, will cause wrong loss silently, didnot know why

            with torch.no_grad():
                gt_heatmap = self.get_gt_heatmap(batch_dict, feature_stride=self.predictor_stride)


        if self.if_save_runtime:
            # merge the submodules save_dict into model_ save_dict
            for cur_module in self.module_list:
                if hasattr(cur_module,"save_dict"):
                    for k_,v_ in cur_module.save_dict.items():
                        if k_ not in self.save_dict:
                            self.save_dict[k_] = []
                        self.save_dict[k_].append(v_)
            if not get_normal_loss:
                # INFO: saving code for predictor
                if 'gt_heatmap' not in self.save_dict:
                    self.save_dict['gt_heatmap'] = []
                self.save_dict['gt_heatmap'].append(gt_heatmap[0].detach().cpu().data)
                if 'pred_heatmap' not in self.save_dict:
                    self.save_dict['pred_heatmap'] = []
                pred_heatmap_d_cpu = {k:v.detach().cpu().data for k,v in pred_heatmap_d.items()}
                self.save_dict['pred_heatmap'].append(pred_heatmap_d_cpu)

        # INFO: merge the inbox_rate dict
        inbox_rate_d = {}
        for k,v in batch_dict.items():
            if '_rate' in k:  # keep inbox_rate and drop_rate both
                inbox_rate_d[k] = v

        # INFO: save some tensors for debugging
        if self.training:

            if get_normal_loss:
                loss, tb_dict, disp_dict = self.get_training_loss()
                ret_dict = {
                        'loss': loss,
                        # 'predictor_loss': loss+self.predictor_loss_lambda*predictor_loss
                        'inbox_rate_d': inbox_rate_d,
                }

                with torch.no_grad():
                    gt_heatmap = self.get_gt_heatmap(batch_dict, feature_stride=self.predictor_stride)

            else:
                predictor_loss, tb_dict, disp_dict = self.get_predictor_loss(pred_heatmap_d, gt_heatmap)
                # DEBUG: when no call get_training_loss, grad accumulation and memory cost increases
                ret_dict = {
                        'predictor_loss': self.predictor_loss_lambda*predictor_loss,
                        # 'predictor_loss': loss+self.predictor_loss_lambda*predictor_loss
                        'inbox_rate_d': inbox_rate_d,
                }



            return ret_dict, tb_dict, disp_dict


        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return pred_dicts, recall_dicts, inbox_rate_d

    def get_gt_heatmap(self, batch_dict, feature_stride=1):
        # batch_dict['spatial_features_2d'].shape # [BS, C, 400, 352]
        # batch_dict['spatial_features_2d_stride']: None
        # batch_dict['spatial_features_stride']: 8
        if hasattr(self.model_cfg.PREDICTOR,'MIN_RADIUS'):
            min_radius = self.model_cfg.PREDICTOR.MIN_RADIUS
        else:
            min_radius = self.model_cfg.TARGET_ASSIGNER.MIN_RADIUS
        predictor_feature_map_size =(self.grid_size[:2]//feature_stride)[[1,0]].astype(int) # INFO: at the beginning of assign_targets, there would be a size [::-1]
        dense_head_clone = self.module_list[4]
        target_dict = dense_head_clone.assign_targets(
                batch_dict['gt_boxes'],
                feature_map_size=predictor_feature_map_size,
                feature_map_stride=self.predictor_stride,
                min_radius=min_radius,
                )

        # torch.save(target_dict['heatmaps'], 'debug_gt_heatmap.pth')
        # import ipdb; ipdb.set_trace()

        # target_dict = self.module_list[4].assign_targets(batch_dict['gt_boxes'],feature_map_size=predictor_feature_map_size.cpu().tolist())
        # print(target_dict['heatmaps'][0].shape)
        return target_dict['heatmaps']

    def get_predictor_loss(self, pred_heatmaps, gt_heatmap):
        # CFG: class agnostic heatmap
        # INFO: gaussian_focal does not support CLASS_AGNOSITIC_HEATMAP=True

        # CLASS_AGNOSITIC_HEATMAP = False
        # if CLASS_AGNOSITIC_HEATMAP:
            # gt_heatmap_ = gt_heatmap[0].sum(dim=1, keepdim=True)  # DEBUG: gt_heatmap is a single element list
        # else:
            # pass

        # CFG: the predictor loss_type
        if self.model_cfg.PREDICTOR.get('LOSS_TYPE',None) is not None:
            PREDICTOR_LOSS_TYPE = self.model_cfg.PREDICTOR.LOSS_TYPE
        else:
            PREDICTOR_LOSS_TYPE = 'mse'

        PREDICTOR_LOSS_TYPE = 'mse'
        if PREDICTOR_LOSS_TYPE == 'mse':
            predictor_loss_func = nn.MSELoss()
        elif PREDICTOR_LOSS_TYPE == 'gaussian_focal':
            predictor_loss_func = loss_utils.FocalLossCenterNet()

        # gt_heatmap_ = gt_heatmap[0]
        gt_heatmap_ = torch.cat([gh.sum(1,keepdim=True) for gh in gt_heatmap],dim=1).sum(1,keepdim=True)
        # torch.save(gt_heatmap_, 'nusc_gt_heatmap.pth')

        predictor_loss = 0.
        for k_, v_ in pred_heatmaps.items():
            # if CLASS_AGNOSITIC_HEATMAP:
                # v_ = v_.sum(dim=1,keepdim=True)
            predictor_loss += predictor_loss_func(v_, gt_heatmap_)

        return predictor_loss, {}, {}   # DBEUG: return empty tb_dict and disp_dict for now



