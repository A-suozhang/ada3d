import pickle
import time

import re

import numpy as np
import torch
import torch.nn as nn
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

from train_utils.train_utils import save_intermediates


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def get_flops_and_memory_opt(cfg, model_,logger,model_type='centerpoint_kitti'):
    # =============== get the flops and mem optimization after hard_drop ================
    # 3d backbone data reduction rate
    # Memory: sum of all activation size (sparse)
    # FLOPS: C_in*C_out*K
    #   - each block consists of 1 sparse conv and 3 submanifold convs
    pre_act_size = 0.
    post_act_size = 0.
    pre_flops = 0.
    post_flops = 0.
    indice_dict_keys = ['subm1','spconv2','spconv3','spconv4']
    channels = [16, 32, 64, 128, 128]
    sparse_rates = []

    model_type = 'centerpoint_kitti'
    # model_type = 'centerpoint_nuscenes'
    if model_type == 'centerpoint_nuscenes':
        block_ids = [[1,4,7,10,13,16],[1,4,7,10,13,16]]
        deblock_ids = [[0],[0]]
    elif model_type == 'centerpoint_kitti':
        block_ids = [[1,4,7,10,13,16],]
        deblock_ids = [[0]]
    else:
        import ipdb; ipdb.set_trace()

    # deblock_ids_plus_one = ['deblock0_{}'.format(_) for _ in [_+1 for _ in [1]]]
    pre_act_size_2d = 0.
    post_act_size_2d = 0.
    pre_flops_2d = 0.
    post_flops_2d = 0.

    for k_,v_ in model_.save_dict.items():
        if '3d' in k_:
            # 'input_pre' and 'out_pre', others are paired 
            if 'input' in k_:   # conv_input_pre
                pre_act_size += v_[0].features.nelement()
                post_act_size += v_[0].features.nelement()
                print(v_[0].indice_dict.keys())
                # get the relative sparse rate with actual data
                for indice_k in indice_dict_keys:
                    mapping = v_[0].indice_dict[indice_k][2][0,:,:]
                    mapping_sparse_rate = (mapping!=-1).sum() / mapping.numel()
                    print('At {}: sparse_rate {:.3f}'.format(indice_k,mapping_sparse_rate))
                    sparse_rates.append(mapping_sparse_rate)
            else:
                if 'out' not in k_ and 'boxes' not in k_:
                    try:
                        idx_ = int(re.search(r'conv_(\d+)', k_).group(1))-1
                    except AttributeError:
                        print(k_)
                        import ipdb; ipdb.set_trace()
                if 'pre' in k_:
                    print(k_, v_[0].features.shape)
                    # print(idx_)
                    pre_flops += 6*channels[idx_]*channels[idx_+1]*sparse_rates[idx_]*27*v_[0].features.shape[0]  # 3 layers, FLOPs=2*C_in*C_out*N
                    pre_act_size += v_[0].features.nelement()
                elif 'post' in k_:
                    print(k_, v_[0].features.shape)
                    # print(idx_)
                    post_flops += 6*channels[idx_]*channels[idx_+1]*sparse_rates[idx_]*27*v_[0].features.shape[0]
                    post_act_size += v_[0].features.nelement()

    backbone_2d_func = model_.backbone_2d
    for i_block, block_ in enumerate(block_ids):
        for i_module in block_:
            save_dict_k = 'block{}_{}'.format(i_block,i_module)
            act = model_.save_dict[save_dict_k]
            op = backbone_2d_func.blocks[i_block][i_module]
            assert isinstance(op, nn.Conv2d)
            in_channels = op.in_channels
            out_channels = op.out_channels
            kernel_k = op.kernel_size[0]*op.kernel_size[1]
            n_nonezero = (act[0]!=0).float().sum()
            n_dense = act[0].numel()
            print('{}: {:.3f}'.format(save_dict_k, (n_nonezero/n_dense).item()))

            pre_act_size_2d += n_dense*out_channels 
            post_act_size_2d += n_nonezero*out_channels
            pre_flops_2d += 2*in_channels*out_channels*kernel_k*n_dense
            post_flops_2d += 2*in_channels*out_channels*kernel_k*n_nonezero

    for i_block, block_ in enumerate(deblock_ids):
        for i_module in block_:
            save_dict_k = 'deblock{}_{}'.format(i_block,i_module+1)
            act = model_.save_dict[save_dict_k]
            op = backbone_2d_func.deblocks[i_block][i_module]
            assert isinstance(op, nn.ConvTranspose2d)
            in_channels = op.in_channels
            out_channels = op.out_channels
            kernel_k = op.kernel_size[0]*op.kernel_size[1]
            n_nonezero = (act[0]!=0).float().sum()
            n_dense = act[0].numel()
            print('{}: {:.3f}'.format(save_dict_k, (n_nonezero/n_dense).item()))

            pre_act_size_2d += n_dense*out_channels
            post_act_size_2d += n_nonezero*out_channels
            pre_flops_2d += 2*in_channels*out_channels*kernel_k*n_dense
            post_flops_2d += 2*in_channels*out_channels*kernel_k*n_nonezero

    logger.info('========= [3D Part] ===========')
    logger.info('==> pre_act_size: {}, post_act_size: {}, Memory Opt: {:.3f}'.format(pre_act_size,post_act_size,post_act_size/pre_act_size))
    logger.info('==> pre_flops: {}, post_flops: {}, FLOPs Opt: {:.3f}'.format(pre_flops/1E9, post_flops/1E9, post_flops/pre_flops))
    logger.info('========= [2D Part] ===========')
    logger.info('==> pre_act_size: {}, post_act_size: {}, Memory Opt: {:.3f}'.format(pre_act_size_2d,post_act_size_2d,post_act_size_2d/pre_act_size_2d))
    logger.info('==> pre_flops: {}, post_flops: {}, FLOPs Opt: {:.3f}'.format(pre_flops_2d/1E9, post_flops_2d/1E9, post_flops_2d/pre_flops_2d))
    logger.info('========= [Overall Opt] ===========')
    mem_opt = (pre_act_size_2d+pre_act_size)/(post_act_size_2d+post_act_size)
    flops_opt = (pre_flops_2d+pre_flops)/(post_flops_2d+post_flops)
    logger.info('==> Mem_opt: {:.3f}, FLOPs_opt: {:.3f}'.format(mem_opt, flops_opt))


def eval_one_epoch(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if args.save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

    if getattr(args, 'infer_time', False):
        start_iter = int(len(dataloader) * 0.1)
        infer_time_meter = common_utils.AverageMeter()

    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()

    if cfg is not None:
        # INFO: get the W,H for predictor, feed the cfg into predictor(in case the hard_drop mode)
        model_ = model.module if hasattr(model,'module') else model
        if hasattr(cfg.OPTIMIZATION,"PREDICTOR"):
            grid_size = (np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3:6])-np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0:3]))/np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[-1]['VOXEL_SIZE'])
            model_.grid_size = grid_size
            # DIRTY: module.cfg = cfg
            model_.backbone_3d.cfg = cfg
            model_.backbone_2d.cfg = cfg
            logger.info("====> grid size for predictor:{}".format(grid_size//model_.predictor_stride))


    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()

    inbox_rate_d = {}
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        # when evaluating ckpt while training, the arg donot have 'export'

        # INFO: for the first batch of data, probe_feature and calc speedup rate
        if i == 0:
            if_save_runtime = True
            model_ = model.module if hasattr(model,'module') else model
            model_.backbone_3d.probe_feature=True
            model_.backbone_2d.probe_feature=True
            if hasattr(model_,'if_save_runtime'): # skip for models without the 'if_save_runtime' flag
                model_.if_save_runtime = if_save_runtime

            with torch.no_grad():
                if getattr(args, 'hard_drop', False):

                    # forward with no hard_drop and save the 3d pre voxels
                    _, _ = model(batch_dict)
                    pre_drop_3d = {}
                    for k,v, in model_.save_dict.items():
                        if '3d' in k and 'pre' in k:
                            pre_drop_3d[k] = v
                    model_.save_dict = {}

                    # forward with hard drop and save the post drop features
                    pred_dicts, recall_dict, inbox_dict = model_.predictor_forward(batch_dict, hard_drop=True)

                    # display the inbox rate
                    inbox_rate_d_ = inbox_dict
                    # merge the inox_rate_d
                    for k,v in inbox_rate_d_.items():
                        if k not in inbox_rate_d.keys():
                            inbox_rate_d[k] = []
                        inbox_rate_d[k].append(v)


                    logger_str =''
                    for k,v in inbox_rate_d.items():
                        sparse_rate_mean_ = np.array(inbox_rate_d[k]).mean()
                        logger_str = logger_str+'\n {}: {:.3f}, '.format(k,sparse_rate_mean_)
                    logger.info(logger_str)
                    inbox_rate_d = {}

                    for k,v in pre_drop_3d.items():
                        print('replace model.save_dict[{}] with no_drop voxels'.format(k))
                        if k=="3d_conv_out_pre":  # dirty_fix, will replace the final 3d_conv_out_pre
                            model_.save_dict['3d_conv_out_post'] = model_.save_dict[k]
                            model_.save_dict[k] = v
                        else:
                            model_.save_dict[k] = v

                    get_flops_and_memory_opt(cfg, model_, logger)

                    if hasattr(args, 'export'):
                        if args.export:
                            # -------------------------------------------------------------------
                            # INFO: the export cfg for hardware_exp
                            # CFG: whether skip eval and apply export_acts
                            logger.info('------- Starting Exporting and Skipping Evalution ---------')
                            model_.save_dict['model_3d'] = model_.backbone_3d
                            # process maskednorm2d to normal nn.BatchNorm2d
                            from pcdet.models.backbones_2d.masked_bn import MaskedBatchNorm2d, MaskedBatchNorm2dV2
                            import torch.nn as nn
                            backbone_2d_func = model_.backbone_2d
                            for i_1, block_ in enumerate(backbone_2d_func.blocks):
                                for i_2, module_ in enumerate(block_):
                                    if isinstance(module_,MaskedBatchNorm2d):
                                        backbone_2d_func.blocks[i_1][i_2] = nn.BatchNorm2d(module_.C)
                                    elif isinstance(module_,MaskedBatchNorm2dV2):
                                        backbone_2d_func.blocks[i_1][i_2] = nn.BatchNorm2d(module_.num_features)
                            for i_1, deblock_ in enumerate(backbone_2d_func.deblocks):
                                for i_2, module_ in enumerate(deblock_):
                                    if isinstance(module_,MaskedBatchNorm2d):
                                        backbone_2d_func.deblocks[i_1][i_2] = nn.BatchNorm2d(module_.C)
                                    elif isinstance(module_,MaskedBatchNorm2dV2):
                                        backbone_2d_func.deblocks[i_1][i_2] = nn.BatchNorm2d(module_.num_features)
                            model_.save_dict['model_2d'] = backbone_2d_func
                            filename_ = final_output_dir.parent.parent / 'exported_intermediates.pth'
                            logger.info('------- Exported File Saved to: -------------\n {}\n ------------------------------ '.format(filename_))
                            save_intermediates(model_, filename_, dist_train=False)
                            return {}
                            # -------------------------------------------------------------------

                else:
                    # only normal forward, if not hard_drop
                    pred_dicts, ret_dict = model(batch_dict)

        else:
            model_.backbone_3d.probe_feature = False
            model_.backbone_2d.probe_feature = False

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        inbox_rate_d = {}
        with torch.no_grad():
            if getattr(args, 'hard_drop', False):

                # --------- debug and probing for density_based -----------
                # debug_only
                # model_.model_cfg.PREDICTOR.DROP_SCHEME = 'density'
                # ------------------------------------------------------
                pred_dicts, recall_dict, inbox_dict = model_.predictor_forward(batch_dict, hard_drop=True)

                inbox_rate_d_ = inbox_dict
                for k,v in inbox_rate_d_.items():
                    if k not in inbox_rate_d.keys():
                        inbox_rate_d[k] = []
                    inbox_rate_d[k].append(v)

            else:
                pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}
        disp_dict['frame_id'] = batch_dict['frame_id']

        if getattr(args, 'infer_time', False):
            inference_time = time.time() - start_time
            infer_time_meter.update(inference_time * 1000)
            # use ms to measure inference time
            disp_dict['infer_time'] = f'{infer_time_meter.val:.2f}({infer_time_meter.avg:.2f})'

        # statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if args.save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()

    if getattr(args,"hard_drop", False):
        # INFO: logger for sparse_rate
        logger_str =''
        for k,v in inbox_rate_d.items():
            sparse_rate_mean_ = np.array(inbox_rate_d[k]).mean()
            logger_str = logger_str+'\n {}: {:.3f}, '.format(k,sparse_rate_mean_)
        logger.info(logger_str)
        inbox_rate_d = {}

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # INFO: display some part of the result_dict
    if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
        # wheh training, args have no namespace test split
        if hasattr(args,'test_split'):
            if not args.test_split:
                result_str_lines = result_str.split('\n')
                logger.info('\n ------ AP Results ---------')
                logger.info('Car: '+result_str_lines[3])
                logger.info('Ped: '+result_str_lines[23])
                logger.info('Cyc: '+result_str_lines[43])
        else:
            result_str_lines = result_str.split('\n')
            logger.info('\n ------ AP Results ---------')
            logger.info('Car: '+result_str_lines[3])
            logger.info('Ped: '+result_str_lines[23])
            logger.info('Cyc: '+result_str_lines[43])
    elif cfg.DATA_CONFIG.DATASET == 'NuScenesDataset':
        # no neetd for extra print
        pass
    else:
        raise NotImplementedError

    logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
