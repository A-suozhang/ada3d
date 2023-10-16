from functools import partial

import torch
import torch.nn as nn
import numpy as np

from SparseBEVTools import points_in_boxes_bev


def check_inbox(cfg, batch_dict, pred_heatmap_, pred_heatmap_valid_mask, if_save_runtime=False):

    # valid_mask: [B,H,W] - [B,1600,1408]
    # point_cloud_range - [1408,1600]
    # gt_boxes: [B,N,8]
    # H=1600 W=1408
    device_ = pred_heatmap_valid_mask.device
    point_cloud_range = torch.tensor(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, device=device_)
    gt_boxes = batch_dict['gt_boxes']   # [B.N,8]
    voxel_size = torch.tensor(cfg.DATA_CONFIG.DATA_PROCESSOR[2].VOXEL_SIZE, device=device_)

    B,H,W = pred_heatmap_valid_mask.shape
    W_, H_, Z_ = ((point_cloud_range[3:]-point_cloud_range[:3])/voxel_size).int()
    feature_stride = W_ // W
    assert W_%W==0 and H_%H==0

    N = H*W
    pred_heatmap_ = pred_heatmap_.permute([0,2,1])
    pred_heatmap_valid_mask = pred_heatmap_valid_mask.permute([0,2,1])
    sparse_rate = lambda x: 1-(torch.sum(x == 0) / x.numel())
    # print(sparse_rate(pred_heatmap_), sparse_rate(pred_heatmap_valid_mask))

    n_drop = 0
    n_point_inbox = 0
    for _ in range(B):
        pre_indices = pred_heatmap_[_,:,:].nonzero()*feature_stride*voxel_size[:2] + point_cloud_range[:2] # leave the batch dim
        post_indices = (pred_heatmap_valid_mask[_,:,:].nonzero()*feature_stride*voxel_size[:2]+point_cloud_range[:2]) # leave the batch dim
        pre_indices = torch.cat([pre_indices, torch.zeros([pre_indices.shape[0],1],device=device_), torch.ones([pre_indices.shape[0],1], device=device_)],dim=1)
        post_indices = torch.cat([post_indices, torch.zeros([post_indices.shape[0],1],device=device_), torch.ones([post_indices.shape[0],1], device=device_)],dim=1)

        mask_pre = points_in_boxes_bev(pre_indices, gt_boxes[_])
        mask_post = points_in_boxes_bev(post_indices, gt_boxes[_])
        n_mask_pre_inbox = mask_pre.nonzero().shape[0]
        n_mask_post_inbox = mask_post.nonzero().shape[0]
        n_point_inbox += n_mask_pre_inbox
        n_drop += (n_mask_pre_inbox - n_mask_post_inbox)

    # how many dropped data is inbox
    if n_point_inbox == 0:
        print('Some boxes points are all dropped, this will cause perf loss, also skip saving')
        if_save_runtime = False
        inbox_rate = 0.
    else:
        inbox_rate = (n_drop / n_point_inbox)

    # if_save_runtime = True   # Vis_only
    if if_save_runtime:
        # ---------------------------------------------------------------------------------
        # INFO: generate grid_map and get masks for visualization
        steps_x = (point_cloud_range[3] - point_cloud_range[0])//(voxel_size[0]*feature_stride)
        steps_y = (point_cloud_range[4] - point_cloud_range[1])//(voxel_size[1]*feature_stride)
        x_grid = torch.linspace(point_cloud_range[0],point_cloud_range[3],int(steps_x), device=device_)
        y_grid = torch.linspace(point_cloud_range[1],point_cloud_range[4],int(steps_y), device=device_)
        xy_meshgrid = torch.stack(torch.meshgrid(x_grid, y_grid),dim=0).reshape([2,-1]).permute([1,0])   # [B,1408,1600] -> [2,N]
        xy_meshgrid = torch.cat([xy_meshgrid,torch.zeros([N,1],device=device_),torch.ones([N,1],device=device_)], dim=1)  # [N,4]
        masks = []
        for _ in range(B):
            mask_ = points_in_boxes_bev(xy_meshgrid, gt_boxes[_])
            masks.append(mask_)
        masks = torch.stack(masks,dim=0).reshape(B,-1,W,H)  # [B,n_box,H,W]
        try:
            masks = (masks.sum(1) != 0).float()
        except:
            import ipdb; ipdb.set_trace()
        sparse_rate = lambda x: (torch.sum(x == 0) / x.numel())

        # pred_heatmap_valid_mask = pred_heatmap_valid_mask.permute([0,2,1])
        # inbox_rate = ((pred_heatmap_valid_mask*masks != 0).sum() / (masks != 0).sum()).item()
        # print('{:.3f}% of inbox pixels remains'.format(inbox_rate*100.0))

        # ---------------------------------------------------------------------------------
        # INFO: save the heatmaps
        save_d = {}
        # save_d['pred_heatmap'] = (pred_heatmap_).detach().cpu().data
        # save_d['masks'] = (masks == 0).float().detach().cpu().data
        # save_d['pred_mask'] = (pred_heatmap_valid_mask).detach().cpu().data
        # save_d['dropped'] = ((pred_heatmap_valid_mask == 0)*(pred_heatmap_ != 0)).detach().cpu().data
        # save_d['inbox_rate'] = float(inbox_rate)
        # INFO: save the point cloud also for visualization
        # save_d['voxel_coords'] = batch_dict['voxel_coords']
        # # process gt_boxes for 3d plot
        # gt_boxes = batch_dict['gt_boxes']
        # center_x = (gt_boxes[:,:,0] - point_cloud_range[0])/voxel_size[0]
        # center_y = (gt_boxes[:,:,1] - point_cloud_range[1])/voxel_size[1]
        # center_z = (gt_boxes[:,:,2] - point_cloud_range[2])/voxel_size[2]
        # len_x = gt_boxes[:,:,3]/voxel_size[0]
        # len_y = gt_boxes[:,:,4]/voxel_size[1]
        # len_z = gt_boxes[:,:,5]/voxel_size[2]
        # rotation = gt_boxes[:,:,6]
        # l_gt_boxes = []
        # for i_batch in range(gt_boxes.shape[0]):
            # label = gt_boxes[i_batch,:,7]
            # nonzero_box_idx = torch.nonzero(label.int()!=0).squeeze()
            # center_x_ = torch.index_select(center_x[i_batch],0,nonzero_box_idx)
            # center_y_ = torch.index_select(center_y[i_batch],0,nonzero_box_idx)
            # center_z_ = torch.index_select(center_z[i_batch],0,nonzero_box_idx)
            # len_x_ = torch.index_select(len_x[i_batch],0,nonzero_box_idx)
            # len_y_ = torch.index_select(len_y[i_batch],0,nonzero_box_idx)
            # len_z_ = torch.index_select(len_z[i_batch],0,nonzero_box_idx)
            # rotation_ = - torch.index_select(rotation[i_batch],0,nonzero_box_idx)
            # box_ = torch.stack([center_z_,center_y_,center_x_,rotation_,len_z_,len_y_,len_x_],dim=1)
            # l_gt_boxes.append(box_)
        # save_d['gt_boxes'] = l_gt_boxes
        # save_d['feature_stride'] = feature_stride
        # torch.save(save_d,'./debug_inbox_3d.pth')
        # import ipdb; ipdb.set_trace()
        # # print('dropped feature inbox (lower the better): {:.2f}'.format(inbox_rate))
        return inbox_rate, save_d
    else:
        return inbox_rate, {}

def probe_drop(pred_heatmap_, pred_heatmap_valid_mask, masks):
    save_d = {}
    save_d['pred_heatmap'] = pred_heatmap_
    save_d['masks'] = (masks == 0)
    save_d['pred_mask'] = pred_heatmap_valid_mask
    save_d['dropped'] = (pred_heatmap_valid_mask == 0)*(pred_heatmap_ != 0)
    torch.save(save_d, 'debug_inbox_3.pth')

def get_topk_mask(importance_score_, k_percent, nonzero_only=True):
    # only get the nonzero element topk
    if nonzero_only:
        B,C,W,H = importance_score_.shape
        pred_heatmap_valid_mask = torch.zeros_like(importance_score_).reshape([B,C,-1])
        # assert importance_score_.min() >= 0
        # INFO: the min of importance score should be 0, so just topk the positive elements
        # but if use density, not true
        for _ in range(B):
            importance_score_cur_batch_flatten = importance_score_[_,:,:,:].view(-1)
            n_nonzero = importance_score_cur_batch_flatten.nonzero().shape[0]  # [N,1]
            num_element = int(k_percent/100.0*n_nonzero)
            topk_values, topk_indices = torch.topk(importance_score_cur_batch_flatten, k=num_element)
            pred_heatmap_valid_mask[_,:,topk_indices] = 1.
        pred_heatmap_valid_mask = pred_heatmap_valid_mask.view_as(importance_score_)
    else:
        importance_score_flatten = importance_score_.view(-1)
        num_element = int(k_percent/100.0*importance_score_flatten.numel())
        topk_values, topk_indices = torch.topk(importance_score_flatten, k=num_element)
        pred_heatmap_valid_mask = torch.zeros_like(importance_score_flatten)
        pred_heatmap_valid_mask[topk_indices] = 1.
        pred_heatmap_valid_mask = pred_heatmap_valid_mask.view_as(importance_score_)

    sparse_rate = lambda x: 1-(torch.sum(x == 0) / x.numel())
    # INFO: print and check the input/output sparse_rate
    # print("pre_drop dense rate {:.4f};  post_drop dense rate {:.4f}".format(sparse_rate(importance_score_),sparse_rate(pred_heatmap_valid_mask)))
    return pred_heatmap_valid_mask, sparse_rate(importance_score_), sparse_rate(pred_heatmap_valid_mask)


