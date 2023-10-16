import os

import torch
import tqdm
import time
import pickle
import glob
import numpy as np

from torch.nn.utils import clip_grad_norm_
from pcdet.utils import common_utils, commu_utils
from pcdet.models import load_data_to_gpu

from train_utils.optimization import build_optimizer, build_scheduler

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_interval=5, ckpt_save_time_interval=300, show_gpu_stat=False, use_amp=False, \
                    cfg=None, hard_drop=False):

    if hard_drop:
        logger.info("=====> [Train-Drop] at epoch {}, finished warmup, with hard drop =======".format(cur_epoch))
    else:
        logger.info("=====> [Train] at epoch {}, in warmup, no hard drop =======".format(cur_epoch))

    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))

    if cfg is not None:
        # INFO: get the W,H for predictor, feed the cfg into predictor(in case the hard_drop mode)
        model_ = model.module if hasattr(model,'module') else model
        grid_size = (np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3:6])-np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0:3]))/np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[-1]['VOXEL_SIZE'])
        model_.grid_size = grid_size
        # DIRTY: module.cfg = cfg
        model_.backbone_3d.cfg = cfg
        model_.backbone_2d.cfg = cfg
        logger.info("====> grid size for predictor:{}".format(grid_size//model_.predictor_stride))

    if rank == 0:
        pbar_str = "train(drop)" if hard_drop else 'train'
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc=pbar_str, dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    end = time.time()
    inbox_rate_d = {}
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')
        
        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        if rank == 0: # only activate saving for rank=0 when ddp
            # flag 'if_save_runtime' to note save_every for model, used for save intermediates during model inference
            rank, world_size = common_utils.get_dist_info()
            # SAVE_RUN_TIME_INTERVAL = 1000 // world_size # if an epoch is less than 1000 iter, save once (accumulated_iter=0) each epoch
            if cur_epoch % ckpt_save_interval == 0:
                if cur_it == start_it or cur_it+1 == total_it_each_epoch:
                    if_save_runtime = True
                else:
                    if_save_runtime = False
            else:
                if_save_runtime = False
            # if_save_runtime = True if cur_it == start_it or cur_it + 1 == total_it_each_epoch else False
            model_ = model.module if hasattr(model,'module') else model
            if hasattr(model_,'if_save_runtime'): # skip for models without the 'if_save_runtime' flag
                model_.if_save_runtime = if_save_runtime

        with torch.cuda.amp.autocast(enabled=use_amp):
            if not hard_drop:
                loss, tb_dict, disp_dict = model_func(model, batch)
            else:
                # INFO: unwrap the model_fn_decorator here
                load_data_to_gpu(batch)
                ret_dict, tb_dict, disp_dict = model_.predictor_forward(batch,hard_drop=hard_drop,get_normal_loss=True)
                loss = ret_dict['loss'].mean()
                inbox_rate_d_ = ret_dict['inbox_rate_d']
                # merge the inox_rate_d
                for k,v in inbox_rate_d_.items():
                    if k not in inbox_rate_d.keys():
                        inbox_rate_d[k] = []
                    inbox_rate_d[k].append(v)
                if hasattr(model, 'update_global_step'):
                    model.update_global_step()
                else:
                    model.module.update_global_step()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()

        accumulated_iter += 1
 
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)
            
            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item() , batch_size)
            
            disp_dict.update({
                'loss': loss.item(), 'lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })
            
            if use_logger_to_record:
                # ADD FEATURE: when using logger, also show progress_bar
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                # tbar.set_postfix(disp_dict)


                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:

                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                    logger.info(
                        'Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )

                    logger_str =''
                    for k,v in inbox_rate_d.items():
                        sparse_rate_mean_ = np.array(inbox_rate_d[k]).mean()
                        logger_str = logger_str+'\n {}: {:.3f} '.format(k,sparse_rate_mean_)
                    logger.info(logger_str)
                    inbox_rate_d = {}

                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                # logging the tb_dict items to tensorboard
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

            # NEW: logging to model_save_dict for each iter: {loss,lr}
            if hasattr(model,'module'): # DIRTY: ck ddp or not
                model_ = model.module
            else:
                model_ = model

            if not hasattr(model_,'save_dict'):
                model_.save_dict = {}
            dicts = [disp_dict, tb_dict]
            for dict_ in dicts:
                for key_, val_ in dict_.items():
                    if not key_ in model_.save_dict.keys():
                        model_.save_dict[key_] = []
                    model_.save_dict[key_].append(val_)

            # DISABLE: save intermediate ckpt every {ckpt_save_time_interval} seconds         
            # time_past_this_epoch = pbar.format_dict['elapsed']
            # if time_past_this_epoch // ckpt_save_time_interval >= ckpt_save_cnt:
                # ckpt_name = ckpt_save_dir / 'latest_model'
                # save_checkpoint(
                    # checkpoint_state(model, optimizer, cur_epoch, accumulated_iter), filename=ckpt_name,
                # )
                # logger.info(f'Save latest model to {ckpt_name}')
                # ckpt_save_cnt += 1

    if rank == 0:
        pbar.close()
    return accumulated_iter

# TODO: the predictor traininig code
def train_predictor_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False, 
                    use_logger_to_record=False, logger=None, logger_iter_interval=50, cur_epoch=None, 
                    total_epochs=None, ckpt_save_dir=None, ckpt_save_interval=5, ckpt_save_time_interval=300, show_gpu_stat=False, use_amp=False,cfg=None,hard_drop=False):  # feed in cfg for predictor
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    ckpt_save_cnt = 1
    start_it = accumulated_iter % total_it_each_epoch

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=optim_cfg.get('LOSS_SCALE_FP16', 2.0**16))
    
    if rank == 0:
        predictor_pbar_str = 'predictor_train(drop)' if hard_drop else 'predictor_train'
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc=predictor_pbar_str, dynamic_ncols=True)
        data_time = common_utils.AverageMeter()
        batch_time = common_utils.AverageMeter()
        forward_time = common_utils.AverageMeter()
        losses_m = common_utils.AverageMeter()

    if hasattr(model,'module'): # DIRTY: ck ddp or not
        model_ = model.module
    else:
        model_ = model

    # INFO: get the W,H for predictor, feed the cfg into predictor
    grid_size = (np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3:6])-np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0:3]))/np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[-1]['VOXEL_SIZE'])
    model_.grid_size = grid_size
    # DIRTY: module.cfg = cfg
    model_.backbone_3d.cfg = cfg
    model_.backbone_2d.cfg = cfg
    logger.info("====> grid size for predictor:{}".format(grid_size//model_.predictor_stride))

    inbox_rate_d = {}

    end = time.time()
    for cur_it in range(start_it, total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        data_timer = time.time()
        cur_data_time = data_timer - end

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/predictor_learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        if rank == 0: # only activate saving for rank=0 when ddp
            # flag 'if_save_runtime' to note save_every for model, used for save intermediates during model inference
            rank, world_size = common_utils.get_dist_info()
            SAVE_RUN_TIME_INTERVAL = 1000 // world_size # if an epoch is less than 1000 iter, save once (accumulated_iter=0) each epoch
            # if_save_runtime = True if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch else False
            # if_save_runtime = True if cur_it == start_it or cur_it + 1 == total_it_each_epoch else Fals]
            if cur_epoch % ckpt_save_interval == 0:
                if cur_it == start_it or cur_it+1 == total_it_each_epoch:
                    if_save_runtime = True
                else:
                    if_save_runtime = False
            else:
                if_save_runtime = False

            model_ = model.module if hasattr(model,'module') else model
            if hasattr(model_,'if_save_runtime'): # skip for models without the 'if_save_runtime' flag
                model_.if_save_runtime = if_save_runtime

        with torch.cuda.amp.autocast(enabled=use_amp):
            # INFO: unwrap the model_fn_decorator here
            load_data_to_gpu(batch)
            ret_dict, tb_dict, disp_dict = model_.predictor_forward(batch,hard_drop=hard_drop)
            loss = ret_dict['predictor_loss'].mean()
            inbox_rate_d_ = ret_dict['inbox_rate_d']
            # merge the inox_rate_d
            for k,v in inbox_rate_d_.items():
                if k not in inbox_rate_d.keys():
                    inbox_rate_d[k] = []
                inbox_rate_d[k].append(v)
            if hasattr(model, 'update_global_step'):
                model.update_global_step()
            else:
                model.module.update_global_step()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        scaler.step(optimizer)
        scaler.update()

        accumulated_iter += 1
        cur_forward_time = time.time() - data_timer
        cur_batch_time = time.time() - end
        end = time.time()

        # average reduce
        avg_data_time = commu_utils.average_reduce_value(cur_data_time)
        avg_forward_time = commu_utils.average_reduce_value(cur_forward_time)
        avg_batch_time = commu_utils.average_reduce_value(cur_batch_time)

        # log to console and tensorboard
        if rank == 0:
            batch_size = batch.get('batch_size', None)

            data_time.update(avg_data_time)
            forward_time.update(avg_forward_time)
            batch_time.update(avg_batch_time)
            losses_m.update(loss.item() , batch_size)
            
            disp_dict.update({
                'predictor_loss': loss.item(), 'predictor_lr': cur_lr, 'd_time': f'{data_time.val:.2f}({data_time.avg:.2f})',
                'f_time': f'{forward_time.val:.2f}({forward_time.avg:.2f})', 'b_time': f'{batch_time.val:.2f}({batch_time.avg:.2f})'
            })
            
            if use_logger_to_record:
                # ADD FEATURE: when using logger, also show progress_bar
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                # tbar.set_postfix(disp_dict)
                if accumulated_iter % logger_iter_interval == 0 or cur_it == start_it or cur_it + 1 == total_it_each_epoch:

                    trained_time_past_all = tbar.format_dict['elapsed']
                    second_each_iter = pbar.format_dict['elapsed'] / max(cur_it - start_it + 1, 1.0)

                    trained_time_each_epoch = pbar.format_dict['elapsed']
                    remaining_second_each_epoch = second_each_iter * (total_it_each_epoch - cur_it)
                    remaining_second_all = second_each_iter * ((total_epochs - cur_epoch) * total_it_each_epoch - cur_it)

                    logger.info(
                        '[Predictor] Train: {:>4d}/{} ({:>3.0f}%) [{:>4d}/{} ({:>3.0f}%)]  '
                        'Loss: {loss.val:#.4g} ({loss.avg:#.3g})  '
                        'LR: {lr:.3e}  '
                        f'Time cost: {tbar.format_interval(trained_time_each_epoch)}/{tbar.format_interval(remaining_second_each_epoch)} ' 
                        f'[{tbar.format_interval(trained_time_past_all)}/{tbar.format_interval(remaining_second_all)}]  '
                        'Acc_iter {acc_iter:<10d}  '
                        'Data time: {data_time.val:.2f}({data_time.avg:.2f})  '
                        'Forward time: {forward_time.val:.2f}({forward_time.avg:.2f})  '
                        'Batch time: {batch_time.val:.2f}({batch_time.avg:.2f})'.format(
                            cur_epoch+1,total_epochs, 100. * (cur_epoch+1) / total_epochs,
                            cur_it,total_it_each_epoch, 100. * cur_it / total_it_each_epoch,
                            loss=losses_m,
                            lr=cur_lr,
                            acc_iter=accumulated_iter,
                            data_time=data_time,
                            forward_time=forward_time,
                            batch_time=batch_time
                            )
                    )

                    # INFO: logger for sparse_rate
                    logger_str =''
                    for k,v in inbox_rate_d.items():
                        sparse_rate_mean_ = np.array(inbox_rate_d[k]).mean()
                        logger_str = logger_str+'\n {}: {:.3f} '.format(k,sparse_rate_mean_)
                    logger.info(logger_str)
                    inbox_rate_d = {}

                    if show_gpu_stat and accumulated_iter % (3 * logger_iter_interval) == 0:
                        # To show the GPU utilization, please install gpustat through "pip install gpustat"
                        gpu_info = os.popen('gpustat').read()
                        logger.info(gpu_info)
            else:
                pbar.update()
                pbar.set_postfix(dict(total_it=accumulated_iter))
                tbar.set_postfix(disp_dict)
                # tbar.refresh()

            if tb_log is not None:
                # logging the tb_dict items to tensorboard
                tb_log.add_scalar('train/predictor_loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/predictor_learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    tb_log.add_scalar('train/predictor_' + key, val, accumulated_iter)

            # NEW: logging to model_save_dict for each iter: {loss,lr}
            if hasattr(model,'module'): # DIRTY: ck ddp or not
                model_ = model.module
            else:
                model_ = model

            if not hasattr(model_,'save_dict'):
                model_.save_dict = {}
            dicts = [disp_dict, tb_dict]
            for dict_ in dicts:
                for key_, val_ in dict_.items():
                    if not key_ in model_.save_dict.keys():
                        model_.save_dict[key_] = []
                    model_.save_dict[key_].append(val_)

    if rank == 0:
        pbar.close()

    return accumulated_iter



def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
        start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
        lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
        merge_all_iters_to_one_epoch=False, use_amp=False,
        use_logger_to_record=False, logger=None, logger_iter_interval=None, ckpt_save_time_interval=None, show_gpu_stat=False,
        args=None,cfg=None,test_loader=None,last_epoch=None,predictor_args={}): # add args and cfgs

    # unpack a few elements
    if hasattr(cfg.MODEL,'PREDICTOR'):
        assert hasattr(cfg.OPTIMIZATION,'PREDICTOR')
        predictor_optimizer = predictor_args['optimizer']
        predictor_lr_scheduler = predictor_args['lr_scheduler']
        predictor_lr_warmup_scheduler = predictor_args['lr_warmup_scheduler']

    accumulated_iter = int(start_iter)
    predictor_accumulated_iter = 0   # INFO: no resume predictor-training

    # debug_only: test the model 1st
    # import ipdb; ipdb.set_trace()
    # with torch.no_grad():
        # intermediate_eval(cfg,args,model,test_loader,start_epoch,logger,dist_test=False,result_dir=ckpt_save_dir)


    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)

        model_ = model.module if hasattr(model, "module") else model
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            if hasattr(cfg.MODEL,'PREDICTOR'):
                # predictor scheduler
                predictor_cur_epoch = (cur_epoch // cfg.OPTIMIZATION.PREDICTOR.TRAIN_PREDICTOR_EVERY)*cfg.OPTIMIZATION.PREDICTOR.TRAIN_PREDICTOR_EPOCH
                predictor_epochs = (cfg.OPTIMIZATION.NUM_EPOCHS // cfg.OPTIMIZATION.PREDICTOR.TRAIN_PREDICTOR_EVERY)*cfg.OPTIMIZATION.PREDICTOR.TRAIN_PREDICTOR_EPOCH

                # the hard_drop or not
                if hasattr(cfg.OPTIMIZATION.PREDICTOR,'WARMUP_EPOCH'):
                    predictor_warmup_done = predictor_cur_epoch >= cfg.OPTIMIZATION.PREDICTOR.WARMUP_EPOCH
                else:
                    predictor_warmup_done = None

                if predictor_warmup_done:
                    logger.info('--------- Done predictor warmup, start hard_drop ---------')
                    hard_drop = True
                else:
                    hard_drop = False

                if predictor_lr_warmup_scheduler is not None and predictor_cur_epoch < cfg.OPTIMIZATION.PREDICTOR.WARMUP_EPOCH:
                    predictor_cur_scheduler = predictor_lr_warmup_scheduler
                else:
                    predictor_cur_scheduler = predictor_lr_scheduler

                if not hasattr(cfg.OPTIMIZATION,"PREDICTOR_ONLY"):
                    SKIP_WEIGHT_TRAINING = False
                else:
                    SKIP_WEIGHT_TRAINING = cfg.OPTIMIZATION.PREDICTOR_ONLY

                if cfg.OPTIMIZATION.PREDICTOR.get('GRADUAL_INCREASE_K',False):
                    train_progress = (predictor_cur_epoch+1)/predictor_epochs
                    model_.backbone_3d.k_percent = [100-(100-x)*train_progress for x in cfg.MODEL.PREDICTOR.K_PERCENT_3D]
                    model_.backbone_2d.k_percent = [100-(100-x)*train_progress for x in cfg.MODEL.PREDICTOR.K_PERCENT_2D]
                else:
                    pass

                if (cur_epoch+1) % cfg.OPTIMIZATION.PREDICTOR.TRAIN_PREDICTOR_EVERY == 0:  # when cur_epoch==0, no predictor training
                    # INFO: share the accumulated_iter for normal and predictor training
                    # reuse and iter through the dataloader would cost 2x data
                    # normally uses train_loader but not dataloader_iter
                    # TODO: how to deal with accumulated_iter,related with lr_scheduler

                    for m in model_.module_list:
                        for param_ in m.parameters():
                            param_.requires_grad = False

                    for _ in range(cfg.OPTIMIZATION.PREDICTOR.TRAIN_PREDICTOR_EPOCH):
                        predictor_accumulated_iter = train_predictor_one_epoch(
                            model, predictor_optimizer, train_loader, model_func,
                            lr_scheduler=predictor_cur_scheduler,
                            accumulated_iter=predictor_accumulated_iter, optim_cfg=cfg.OPTIMIZATION.PREDICTOR,
                            rank=rank, tbar=tbar, tb_log=tb_log,
                            leave_pbar=(predictor_cur_epoch + 1 == cfg.OPTIMIZATION.PREDICTOR.NUM_EPOCHS),
                            total_it_each_epoch=total_it_each_epoch,
                            dataloader_iter=dataloader_iter, 
                            cur_epoch=predictor_cur_epoch, total_epochs=cfg.OPTIMIZATION.PREDICTOR.NUM_EPOCHS,
                            use_logger_to_record=use_logger_to_record, 
                            logger=logger, logger_iter_interval=logger_iter_interval,
                            ckpt_save_dir=ckpt_save_dir, ckpt_save_interval=ckpt_save_interval, ckpt_save_time_interval=ckpt_save_time_interval, 
                            show_gpu_stat=show_gpu_stat,
                            use_amp=use_amp,
                            cfg=cfg,
                            hard_drop=hard_drop,
                        )
                        predictor_cur_epoch += 1

                    # INFO: recover the backbone requires_grad
                    for m in model_.module_list:
                        for param_ in m.parameters():
                            param_.requires_grad = True

                if not SKIP_WEIGHT_TRAINING:

                    for m in [model_.predictor, model_.predictor_preprocess]:
                        for param_ in m.parameters():
                            param_.requires_grad = False

                    accumulated_iter = train_one_epoch(
                        model, optimizer, train_loader, model_func,
                        lr_scheduler=cur_scheduler,
                        accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                        rank=rank, tbar=tbar, tb_log=tb_log,
                        leave_pbar=(cur_epoch + 1 == total_epochs),
                        total_it_each_epoch=total_it_each_epoch,
                        dataloader_iter=dataloader_iter, 
                        cur_epoch=cur_epoch, total_epochs=total_epochs,
                        use_logger_to_record=use_logger_to_record, 
                        logger=logger, logger_iter_interval=logger_iter_interval,
                        ckpt_save_dir=ckpt_save_dir, ckpt_save_interval=ckpt_save_interval, ckpt_save_time_interval=ckpt_save_time_interval, 
                        show_gpu_stat=show_gpu_stat,
                        use_amp=use_amp,
                        cfg=cfg,
                        hard_drop=hard_drop,
                    )

                    for m in [model_.predictor, model_.predictor_preprocess]:
                        for param_ in m.parameters():
                            param_.requires_grad = True

            else: # normal training
                accumulated_iter = train_one_epoch(
                        model, optimizer, train_loader, model_func,
                        lr_scheduler=cur_scheduler,
                        accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                        rank=rank, tbar=tbar, tb_log=tb_log,
                        leave_pbar=(cur_epoch + 1 == total_epochs),
                        total_it_each_epoch=total_it_each_epoch,
                        dataloader_iter=dataloader_iter, 
                        cur_epoch=cur_epoch, total_epochs=total_epochs,
                        use_logger_to_record=use_logger_to_record, 
                        logger=logger, logger_iter_interval=logger_iter_interval,
                        ckpt_save_dir=ckpt_save_dir, ckpt_save_interval=ckpt_save_interval, ckpt_save_time_interval=ckpt_save_time_interval, 
                        show_gpu_stat=show_gpu_stat,
                        use_amp=use_amp
                    )


            # save trained model
            trained_epoch = cur_epoch + 1

            # note whether ddp or not
            if args.launcher == 'none':
                dist_train = False
            else:
                dist_train = True

            # NEW; when saving the ckpt, eval the model
            EVAL_CKPT_WHILE_SAVING=True
            if EVAL_CKPT_WHILE_SAVING:
                if trained_epoch % ckpt_save_interval == 0:
                    with torch.no_grad():
                        intermediate_eval(cfg,args,model,test_loader,trained_epoch,logger,dist_test=dist_train,result_dir=ckpt_save_dir)

            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )

            # CFG: SAVE_INTERMEDIATES_INTERVAL
            SAVE_INTERMEDIATES_INTERVAL = 1
            # print((trained_epoch-1)%SAVE_INTERMEDIATES_INTERVAL)
            if (trained_epoch-1)%SAVE_INTERMEDIATES_INTERVAL==0 and rank==0:  # save at epoch=0
                # NEW: for each epoch, save intermediates
                filename_=ckpt_save_dir / 'intermediates.pth'
                save_intermediates(
                        model, filename_, dist_train = dist_train
                )
            else:
                # also empty the save_dict when not saving
                model_ = model.module if hasattr(model,'module') else model
                for cur_module in model_.module_list:
                    if hasattr(cur_module, 'save_dict'):
                        cur_module.save_dict = {}



def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        if torch.__version__ >= '1.4':
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename, _use_new_zipfile_serialization=False)
        else:
            torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    if torch.__version__ >= '1.4':
        torch.save(state, filename, _use_new_zipfile_serialization=False)
    else:
        torch.save(state, filename)


def save_intermediates(model, filename=None,dist_train=False):
    # save model.save_dict into pth file
    assert filename is not None
    if dist_train:
        model_ = model.module
    else:
        model_ = model
    if not hasattr(model_,'save_dict'):
        return None

    torch.save(model_.save_dict, filename)
    # clean the model_.modules's sav_dict
    for cur_module in model_.module_list:
        if hasattr(cur_module,'save_dict'):
            cur_module.save_dict = {}

    return None

# ---- copied and modified from ../eval_utils/eval_utils.py -----
def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])

def intermediate_eval(cfg, args, model, dataloader, epoch_id, logger, dist_test=False, result_dir=None):
    # INFO: save the model or evaluate the predictor. 
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
        # the input model is already ddp mode, so no need to ddp wrap it
        # model = torch.nn.parallel.DistributedDataParallel(
                # model,
                # device_ids=[local_rank],
                # broadcast_buffers=False
        # )
    model.eval()

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)

    model_ = model.module if hasattr(model,'module') else model
    if hasattr(model_,'if_save_runtime'): # skip for models without the 'if_save_runtime' flag
        model_.if_save_runtime = False  # dont save_runtime during eval

    grid_size = (np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[3:6])-np.array(cfg.DATA_CONFIG.POINT_CLOUD_RANGE[0:3]))/np.array(cfg.DATA_CONFIG.DATA_PROCESSOR[-1]['VOXEL_SIZE'])
    model_.grid_size = grid_size
    # DIRTY: module.cfg = cfg
    model_.backbone_3d.cfg = cfg
    model_.backbone_2d.cfg = cfg
    if hasattr(model_,'predictor_stride'):
        logger.info("====> grid size for predictor:{}".format(grid_size//model_.predictor_stride))

    start_time = time.time()
    inbox_rate_d = {}
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)

        if getattr(args, 'infer_time', False):
            start_time = time.time()

        with torch.no_grad():
            if hasattr(cfg.OPTIMIZATION,"PREDICTOR"):
                # DEBUG: ddp test of predictor forward raises ddp error
                if dist_test:
                    pred_dicts, recall_dict, inbox_dict = model.module.predictor_forward(batch_dict, hard_drop=True)
                else:
                    pred_dicts, recall_dict, inbox_dict = model.predictor_forward(batch_dict, hard_drop=True)
                inbox_rate_d_ = inbox_dict
                # merge the inox_rate_d
                for k,v in inbox_rate_d_.items():
                    if k not in inbox_rate_d.keys():
                        inbox_rate_d[k] = []
                    inbox_rate_d[k].append(v)
                # _, ret_dict = model(batch_dict)
                # print(inbox_rate_d)
            else:
                pred_dicts, ret_dict = model(batch_dict)

        disp_dict = {}

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

    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir='./tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir='./tmpdir')

    if hasattr(cfg.OPTIMIZATION,"PREDICTOR"):
        # INFO: logger for sparse_rate
        logger_str =''
        for k,v in inbox_rate_d.items():
            sparse_rate_mean_ = np.array(inbox_rate_d[k]).mean()
            logger_str = logger_str+'\n {}: {:.3f}, '.format(k,sparse_rate_mean_)
        logger.info(logger_str)
        inbox_rate_d = {}

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

    # with open(result_dir / 'result.pkl', 'wb') as f:
        # pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    # INFO: display some part of the result_dict
    if cfg.DATA_CONFIG.DATASET == 'KittiDataset':
        result_str_lines = result_str.split('\n')
        logger.info('\n ------ AP Results ---------')
        logger.info('Car: '+result_str_lines[3])
        logger.info('Ped: '+result_str_lines[23])
        logger.info('Cyc: '+result_str_lines[43])
    elif cfg.DATA_CONFIG.DATASET == 'NuScenesDataset':
        # no neetd for extra print
        pass
    else:
        import ipdb; ipdb.set_trace()

    # logger.info('Result is saved to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict

