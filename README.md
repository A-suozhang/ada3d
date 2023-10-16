# Ada3D : Exploiting the Spatial Redundancy with Adaptive Inference for Efficient 3D Object Detection, ICCV23

Authors: [Tianchen Zhao](https://nicsefc.ee.tsinghua.edu.cn/people/TianchenZhao), Xuefei Ning, Ke Hong, Zhongyuan Qiu, Pu Lu, Yali Zhao, Linfeng Zhang, Lipu Zhou, Guohao Dai, Huazhong Yang, Yu Wang

![](https://github.com/A-suozhang/MyPicBed/raw/master/img/20231016224256.png)

This is the implemention of the ICCV23 paper: [Ada3D](https://arxiv.org/abs/2307.08209), For more information, please refer to our Project Page at [https://a-suozhang.xyz/ada3d.github.io/](https://a-suozhang.xyz/ada3d.github.io/)



# Get Started

The code is tested on the environment listed below:

```
torch=1.9.1+cu111
spconv=1.2
openpcdet=0.6.0+633bd6b
CUDA=11.1
RTX3090 GPU. 
```

(1) Setup the Environment :

- install torch (the code is tested on torch v1.9.1+cu111)

```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

- follow the installation guide to install [SPConv-V1.2](https://github.com/traveller59/spconv/tree/v1.2.1)

- local install `sparse_bev_tools`

```
cd SparseBEVTools
python setup.py develop
```

- local install [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md) 

```
pip install -r requirements.txt
python setup.py develop
```

(2) Prepare the Data:

- link the KITTI dataset

```
ln -s $KITTI_PATH ./data/kitti
```

(3) Download the Pre-trained Model:

- Download the Ada3D-B model though [Google Drive](https://drive.google.com/file/d/17qBRnIbr6d2uhg4wWKbk2gLoz5bcfhhD/view?usp=sharing) and put it in `./tools/pretrained-models/`

(More pre-trained models on the way)


# Usage

## Training

0. enter the `./tools` folder

1. train the model with Sparsity-Preserving BN:

```
python train.py --cfg_file ./cfgs/masked_bn.yaml --extra_tag masked_bn
```


(the distributed data parallel training is supported, with the same input args of the command example:)

```
CUDA_VISIBLE_DEVICES=0,1 bash scripts/dist_train.sh 2 --cfg_file ./cfgs/masked_bn.yaml --extra_tag masked_bn
```

2. train the predictor (with the generated model from the 1st step)

```
python train.py --cfg_file ./cfgs/predictor_masked_bn.yaml --extra_tag predictor_train --ckpt ../output/cfgs/masked_bn/default/ckpt/latest_model.pth
```

3. joint fintune the predictor and the backbone (with the generated model from the 2nd step):

```
python train.py --cfg_file ./cfgs/predictor_masked_bn_tune.yaml --extra_tag predictor_tune --ckpt ../output/cfgs/predictor/default/ckpt/latest_model.pth --hard_drop
```

## Testing

- we provide the pre-trained Ada3D-B centerpoint model on KITTI with results listed below:

```
python test.py --cfg ./cfgs/predictor_masked_bn_tune.yaml --ckpt ./pretrained-models/checkpoint.pth --hard_drop
```



# Acknowledgement

This project relies on code and libraries from other open-source projects. We want to express our gratitude to the following developers and projects:

- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet), 
- [CenterPoint-Kitti](https://github.com/tianweiy/CenterPoint-KITTI),


# Citation

If you find our work is useful in your research, please consider citing:

```
@article{Zhao2023Ada3DE,
  title={Ada3D : Exploiting the Spatial Redundancy with Adaptive Inference for Efficient 3D Object Detection},
  author={Tianchen Zhao and Xuefei Ning and Ke Hong and Zhongyuan Qiu and Pu Lu and Yali Zhao and Linfeng Zhang and Lipu Zhou and Guohao Dai and Huazhong Yang and Yu Wang},
  journal={ArXiv},
  year={2023},
  volume={abs/2307.08209},
  url={https://api.semanticscholar.org/CorpusID:259937318}
}
```

# TODO

- [ ] nuScenes and ONCE training script.
- [ ] merge SPVNAS and VoxelNeXT repo.

