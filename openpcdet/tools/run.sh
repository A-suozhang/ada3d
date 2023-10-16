# --------  debug tranining -----------
# CUDA_VISIBLE_DEVICES=$1 python train.py --cfg_file ./cfgs/kitti_models/centerpoint.yaml --extra_tag debug0
# CUDA_VISIBLE_DEVICES=1 python train.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml --extra_tag debug
# CUDA_VISIBLE_DEVICES=$1 python train.py --cfg_file ./cfgs/kitti_models/ablations/drop_voxel.yaml --extra_tag drop_voxel_$2
 #CUDA_VISIBLE_DEVICES=$1 python train.py --cfg_file ./cfgs/kitti_models/ablations/masked_bn.yaml --extra_tag masked_bn_debug
 #CUDA_VISIBLE_DEVICES=$1 python train.py --cfg_file ./cfgs/kitti_models/ablations/predictor.yaml --extra_tag debug


# --------  debug ddp tranining -----------
 # CUDA_VISIBLE_DEVICES=1,2,3,4 bash scripts/dist_train.sh 4 --cfg_file $1 --extra_tag debug_ddp
#CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/dist_train.sh 4 --cfg_file ./cfgs/kitti_models/ablations/masked_bn.yaml --extra_tag debug_mask_bn_ddp
#CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/dist_train.sh 4  --cfg ./cfgs/kitti_models/ablations/masked_bn_finetune.yaml  --extra_tag masked_bn_finetune --ckpt ../output/cfgs/kitti_models/centerpoint/default/ckpt/latest_model_for_masked_bn.pth

#echo CUDA_VISIBLE_DEVICES=4,5,6,7 bash scripts/dist_train.sh 4 --cfg ./cfgs/kitti_models/ablations/masked_bn_finetune.yaml  --extra_tag masked_bn_finetune_ddp --ckpt ../output/cfgs/kitti_models/centerpoint/default/ckpt/latest_model_for_masked_bn.pth
#echo CUDA_VISIBLE_DEVICES=5,6 bash scripts/dist_train.sh 2 --cfg ./cfgs/kitti_models/finetune/finetune_centerpoint.yaml  --extra_tag lr3e-4_wd1e-4 --ckpt ../output/cfgs/kitti_models/centerpoint/default/ckpt/latest_modet.pth

# --------  debug test -----------
#LOG_PATH="../output/cfgs/kitti_models/finetune/finetune_centerpoint/test_finetune/"
#LOG_PATH="../output/cfgs/kitti_models/ablations/masked_bn_finetune/masked_bn_finetune/"
#CFG_NAME="masked_bn_finetune.yaml"
##CFG_NAME="finetune_centerpoint.yaml"

#CKPT_NAME="checkpoint_epoch_85.pth"
##CUDA_VISIBLE_DEVICES=$1 python test.py --cfg ${LOG_PATH}${CFG_NAME} --ckpt ${LOG_PATH}ckpt/${CKPT_NAME}

#CUDA_VISIBLE_DEVICES=$1 python test.py --cfg ./cfgs/kitti_models/ablations/masked_bn_finetune.yaml --ckpt ../output/cfgs/kitti_models/ablations/masked_bn_finetune/masked_bn_finetune/ckpt/checkpoint_epoch_80.pth
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file ./cfgs/kitti_models/ablations/masked_bn.yaml --ckpt ../output/cfgs/kitti_models/ablations/masked_bn/masked_bn/ckpt/checkpoint_epoch_80.pth
#CUDA_VISIBLE_DEVICES=1 python test.py --cfg_file ./cfgs/kitti_models/ablations/masked_bn.yaml --ckpt ../output/cfgs/kitti_models/centerpoint/default/ckpt/latest_model_for_masked_bn.pth


# --------  debug ddp test -----------



