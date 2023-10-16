import numpy as np
import math
from typing import Optional, Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.modules.batchnorm import _BatchNorm
from torch.autograd.function import Function


class MaskedBatchNorm2d(nn.Module):
    def __init__(self, C, eps=1e-3, momentum=0.1, affine=False):
        super().__init__()
        # self.bn = nn.BatchNorm2d(C, momentum=1., eps=eps, affine=affine)   # DEBUG_ONLY
        self.C = C
        self.momentum = momentum
        self.eps = eps
        # self.running_mean = 0.
        running_var = torch.ones([C])
        self.register_buffer('running_var',running_var)
        self.training = True
        self.affine = affine
        self.num_batches_tracked = 0.

        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(C))
            # self.affine_transform = nn.Linear(C,C,bias=False)

    def forward(self, x):
        B,C,W,H = x.shape

        # DEBUG_ONLY: the other gather funciton @qzy
        # nonzero_idx = x.abs().sum(1).view(-1,).nonzero()
        # features = x.permute(0,2,3,1).contiguous().view(-1,C)
        # nonzero_features = features[nonzero_idx].view(-1,C)   # [K,C]

        x1=x.sum(1)
        nonzero_idxs = []
        num_nonzeros = []
        for _ in range(B):
            nonzero_idx_cur_batch = x1[_].nonzero()
            num_nonzero = nonzero_idx_cur_batch.shape[0]
            nonzero_idxs.append(nonzero_idx_cur_batch)
            num_nonzeros.append(num_nonzero)
        # print('nonzero-rate', max(num_nonzeros)/(W*H))
        idx_for_gather = torch.zeros([B,max(num_nonzeros),2],device=x.device)
        for _ in range(B):
            idx_for_gather[_,:num_nonzeros[_],:] = nonzero_idxs[_]
        # since different batch has different nonzero elments
        # zero-padded, gathered then replaced with zero(since the upper left element is always empty)
        # Inputs: [B,C,W,J]
        # idxs: [B,K,2]
        # gathered: [B,C,K]
        idx_for_gather = idx_for_gather.long()
        batch_enum = torch.arange(B).unsqueeze(1)
        gathered = x[batch_enum,:,idx_for_gather[:,:,0],idx_for_gather[:,:,1]]
        gathered = gathered.permute([0,2,1]).unsqueeze(-1).contiguous()  # [B,C,K]
        # save_d = {}
        # save_d['pre-bn'] = gathered
        # gathered_bn = self.bn(gathered)

        mean_ = gathered.mean(dim=(0,2,3),keepdim=True)
        if not self.training:
            var_ = self.running_var.reshape([1,-1,1,1])
        else:
            var_ = ((gathered - mean_)**2).mean(dim=(0,2,3), keepdim=True)
            self.running_var = (1.-self.momentum)*self.running_var + self.momentum*var_.view(-1)

        # DEBUG: when substracting mean, it wont work
        gathered_bn = (gathered) / torch.sqrt(var_ + self.eps)  # DEBUG_ONLY!
        # gathered_bn = self.bn(gathered)
        if self.affine:
            gathered_bn = gathered_bn*self.weight.reshape([1,-1,1,1])

        # print((mean_.reshape([C]) - self.bn.running_mean).abs().max())
        # print((var_.reshape([C]) - self.bn.running_var).abs().max())
        # print((gathered_bn_ - gathered_bn).abs().max())

        # save_d['post-bn'] = gathered_bn

        # torch.save(save_d, './visual_utils/probe-bn-temp.pth')
        x[batch_enum,:,idx_for_gather[:,:,0],idx_for_gather[:,:,1]] = gathered_bn.squeeze(-1).permute([0,2,1]).contiguous()
        # x[:,:,0,0] = x[:,:,0,0]*0.   # mask the upper left

        return x

class MaskedBatchNorm2dV2(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1.e-3, momentum=0.1, affine=True):
        super().__init__(num_features, eps=eps, momentum=momentum,affine=affine)
        if self.affine:
            # INFO: delete bias since its a parameter not used in producing loss, otherwise would raise DDP error
            del self.bias
            del self.running_mean

    def forward(self, x):

        B,C,W,H = x.shape
        nonzero_idx = x.abs().sum(1).view(-1,).nonzero()
        features = x.permute(0,2,3,1).contiguous().view(-1,C)
        nonzero_features = features[nonzero_idx]   # [K,C]

        if not self.training:
            var_ = self.running_var.reshape([1,1,C])
        else:
            mean_ = nonzero_features.mean(0,keepdim=True)  # [1,C]
            var_ = ((nonzero_features - mean_)**2).mean(0,keepdim=True)  # [1,C]
            self.running_var = (1.-self.momentum)*self.running_var + self.momentum*var_.reshape([C])

        nonzero_features = nonzero_features / torch.sqrt(var_ + self.eps)
        if self.affine:
            nonzero_features = nonzero_features*self.weight

        dense_output = features.new_zeros(size=features.size(), dtype=torch.float32)
        dense_output[nonzero_idx] = nonzero_features
        dense_output = dense_output.view(B,H,W,C).permute(0,3,1,2).contiguous()

        # DEBUG_ONLY
        # print((dense_output==0).int().sum() / dense_output.nelement())

        return dense_output

class MaskedSyncBatchNormV2(_BatchNorm):
    r"""Applies Batch Normalization over a N-Dimensional input (a mini-batch of [N-2]D inputs
    with additional channel dimension) as described in the paper
    `Batch Normalization: Accelerating Deep Network Training by Reducing
    Internal Covariate Shift <https://arxiv.org/abs/1502.03167>`__ .
    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
    The mean and standard-deviation are calculated per-dimension over all
    mini-batches of the same process groups. :math:`\gamma` and :math:`\beta`
    are learnable parameter vectors of size `C` (where `C` is the input size).
    By default, the elements of :math:`\gamma` are sampled from
    :math:`\mathcal{U}(0, 1)` and the elements of :math:`\beta` are set to 0.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.
    Also by default, during training this layer keeps running estimates of its
    computed mean and variance, which are then used for normalization during
    evaluation. The running estimates are kept with a default :attr:`momentum`
    of 0.1.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and batch statistics are instead used during
    evaluation time as well.
    .. note::
        This :attr:`momentum` argument is different from one used in optimizer
        classes and the conventional notion of momentum. Mathematically, the
        update rule for running statistics here is
        :math:`\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t`,
        where :math:`\hat{x}` is the estimated statistic and :math:`x_t` is the
        new observed value.
    Because the Batch Normalization is done for each channel in the ``C`` dimension, computing
    statistics on ``(N, +)`` slices, it's common terminology to call this Volumetric Batch
    Normalization or Spatio-temporal Batch Normalization.
    Currently :class:`SyncBatchNorm` only supports
    :class:`~torch.nn.DistributedDataParallel` (DDP) with single GPU per process. Use
    :meth:`torch.nn.SyncBatchNorm.convert_sync_batchnorm()` to convert
    :attr:`BatchNorm*D` layer to :class:`SyncBatchNorm` before wrapping
    Network with DDP.
    Args:
        num_features: :math:`C` from an expected input of size
            :math:`(N, C, +)`
        eps: a value added to the denominator for numerical stability.
            Default: ``1e-5``
        momentum: the value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average). Default: 0.1
        affine: a boolean value that when set to ``True``, this module has
            learnable affine parameters. Default: ``True``
        track_running_stats: a boolean value that when set to ``True``, this
            module tracks the running mean and variance, and when set to ``False``,
            this module does not track such statistics, and initializes statistics
            buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
            When these buffers are ``None``, this module always uses batch statistics.
            in both training and eval modes. Default: ``True``
        process_group: synchronization of stats happen within each process group
            individually. Default behavior is synchronization across the whole
            world
    Shape:
        - Input: :math:`(N, C, +)`
        - Output: :math:`(N, C, +)` (same shape as input)
    .. note::
        Synchronization of batchnorm statistics occurs only while training, i.e.
        synchronization is disabled when ``model.eval()`` is set or if
        ``self.training`` is otherwise ``False``.
    Examples::
        >>> # xdoctest: +SKIP
        >>> # With Learnable Parameters
        >>> m = nn.SyncBatchNorm(100)
        >>> # creating process group (optional)
        >>> # ranks is a list of int identifying rank ids.
        >>> ranks = list(range(8))
        >>> r1, r2 = ranks[:4], ranks[4:]
        >>> # Note: every rank calls into new_group for every
        >>> # process group created, even if that rank is not
        >>> # part of the group.
        >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
        >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
        >>> # Without Learnable Parameters
        >>> m = nn.BatchNorm3d(100, affine=False, process_group=process_group)
        >>> input = torch.randn(20, 100, 35, 45, 10)
        >>> output = m(input)
        >>> # network is nn.BatchNorm layer
        >>> sync_bn_network = nn.SyncBatchNorm.convert_sync_batchnorm(network, process_group)
        >>> # only single gpu per process is currently supported
        >>> ddp_sync_bn_network = torch.nn.parallel.DistributedDataParallel(
        >>>                         sync_bn_network,
        >>>                         device_ids=[args.local_rank],
        >>>                         output_device=args.local_rank)
    """

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
        process_group: Optional[Any] = None,
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__(
            num_features, eps, momentum, affine, track_running_stats, **factory_kwargs
        )
        self.process_group = process_group

    def _check_input_dim(self, input):
        if input.dim() < 2:
            raise ValueError(
                "expected at least 2D input (got {}D input)".format(input.dim())
            )

    def _check_non_zero_input_channels(self, input):
        if input.size(1) == 0:
            raise ValueError(
                "SyncBatchNorm number of input channels should be non-zero"
            )

    def forward(self, input: Tensor) -> Tensor:
        self._check_input_dim(input)
        self._check_non_zero_input_channels(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            assert self.num_batches_tracked is not None
            self.num_batches_tracked.add_(1)
            if self.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / self.num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        # If buffers are not to be tracked, ensure that they won't be updated
        running_mean = (
            self.running_mean if not self.training or self.track_running_stats else None
        )
        running_var = (
            self.running_var if not self.training or self.track_running_stats else None
        )

        # Don't sync batchnorm stats in inference mode (model.eval()).
        need_sync = (bn_training and self.training and
                     torch.distributed.is_available() and torch.distributed.is_initialized())
        if need_sync:
            # currently only GPU input is supported
            if not input.is_cuda:
                raise ValueError("SyncBatchNorm expected input tensor to be on GPU")

            process_group = torch.distributed.group.WORLD
            if self.process_group:
                process_group = self.process_group
            world_size = torch.distributed.get_world_size(process_group)
            need_sync = world_size > 1

        # fallback to framework BN when synchronization is not necessary
        if not need_sync:
            return F.batch_norm(
                input,
                running_mean,
                running_var,
                self.weight,
                self.bias,
                bn_training,
                exponential_average_factor,
                self.eps,
            )
        else:
            assert bn_training
            return sync_masked_batch_norm.apply(  # INFO: change to sync_masked_batch_norm
                input,
                self.weight,
                self.bias,
                running_mean,
                running_var,
                self.eps,
                exponential_average_factor,
                process_group,
                world_size,
            )

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Helper function to convert all :attr:`BatchNorm*D` layers in the model to
        :class:`torch.nn.SyncBatchNorm` layers.
        Args:
            module (nn.Module): module containing one or more :attr:`BatchNorm*D` layers
            process_group (optional): process group to scope synchronization,
                default is the whole world
        Returns:
            The original :attr:`module` with the converted :class:`torch.nn.SyncBatchNorm`
            layers. If the original :attr:`module` is a :attr:`BatchNorm*D` layer,
            a new :class:`torch.nn.SyncBatchNorm` layer object will be returned
            instead.
        Example::
            >>> # Network with nn.BatchNorm layer
            >>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA)
            >>> module = torch.nn.Sequential(
            >>>            torch.nn.Linear(20, 100),
            >>>            torch.nn.BatchNorm1d(100),
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # ranks is a list of int identifying rank ids.
            >>> ranks = list(range(8))
            >>> r1, r2 = ranks[:4], ranks[4:]
            >>> # Note: every rank calls into new_group for every
            >>> # process group created, even if that rank is not
            >>> # part of the group.
            >>> # xdoctest: +SKIP("distributed")
            >>> process_groups = [torch.distributed.new_group(pids) for pids in [r1, r2]]
            >>> process_group = process_groups[0 if dist.get_rank() <= 3 else 1]
            >>> sync_bn_module = torch.nn.SyncBatchNorm.convert_sync_batchnorm(module, process_group)
        """
        module_output = module
        if isinstance(module, MaskedBatchNorm2dV2):
            module_output = MaskedSyncBatchNormV2(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
            print('converting masked_bn to sync_masked_bn')

        elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module_output = torch.nn.SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group,
            )
            if module.affine:
                with torch.no_grad():
                    module_output.weight = module.weight
                    module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.qconfig = module.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output


class sync_masked_batch_norm(Function):

    @staticmethod
    def forward(self, input, weight, bias, running_mean, running_var, eps, momentum, process_group, world_size):
        if not (
            input.is_contiguous(memory_format=torch.channels_last) or
            input.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            input = input.contiguous()
        if weight is not None:
            weight = weight.contiguous()

        size = int(input.numel() // input.size(1))
        if size == 1 and world_size < 2:
            raise ValueError('Expected more than 1 value per channel when training, got input size {}'.format(size))

        num_channels = input.shape[1]
        if input.numel() > 0:
            # calculate mean/invstd for input.
            mean, invstd = torch.batch_norm_stats(input, eps)
			# INFO: get_dummy_mean
            dummy_mean = mean.new_zeros(size=mean.size(), dtype=mean.dtype, device=mean.device)
            mean = dummy_mean

            count = torch.full(
                (1,),
                input.numel() // input.size(1),
                dtype=mean.dtype,
                device=mean.device
            )

            # C, C, 1 -> (2C + 1)
            combined = torch.cat([mean, invstd, count], dim=0)
        else:
            # for empty input, set stats and the count to zero. The stats with
            # zero count will be filtered out later when computing global mean
            # & invstd, but they still needs to participate the all_gather
            # collective communication to unblock other peer processes.
            combined = torch.zeros(
                2 * num_channels + 1,
                dtype=input.dtype,
                device=input.device
            )

        # Use allgather instead of allreduce because count could be different across
        # ranks, simple all reduce op can not give correct results.
        # batch_norm_gather_stats_with_counts calculates global mean & invstd based on
        # all gathered mean, invstd and count.
        # for nccl backend, use the optimized version of all gather.
        if process_group._get_backend_name() == 'nccl':
            # world_size * (2C + 1)
            combined_size = combined.numel()
            combined_flat = torch.empty(1,
                                        combined_size * world_size,
                                        dtype=combined.dtype,
                                        device=combined.device)
            dist.all_gather_into_tensor(combined_flat, combined, process_group, async_op=False)
            combined = torch.reshape(combined_flat, (world_size, combined_size))
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)
        else:
            # world_size * (2C + 1)
            combined_list = [
                torch.empty_like(combined) for _ in range(world_size)
            ]
            dist.all_gather(combined_list, combined, process_group, async_op=False)
            combined = torch.stack(combined_list, dim=0)
            # world_size * (2C + 1) -> world_size * C, world_size * C, world_size * 1
            mean_all, invstd_all, count_all = torch.split(combined, num_channels, dim=1)

        if not torch.cuda.is_current_stream_capturing():
            # The lines below force a synchronization between CUDA and CPU, because
            # the shape of the result count_all depends on the values in mask tensor.
            # Such synchronizations break CUDA Graph capturing.
            # See https://github.com/pytorch/pytorch/issues/78549
            # FIXME: https://github.com/pytorch/pytorch/issues/78656 describes
            # a better longer-term solution.

            # remove stats from empty inputs
            mask = count_all.squeeze(-1) >= 1
            count_all = count_all[mask]
            mean_all = mean_all[mask]
            invstd_all = invstd_all[mask]

        # calculate global mean & invstd
        mean, invstd = torch.batch_norm_gather_stats_with_counts(
            input,
            mean_all,
            invstd_all,
            running_mean,
            running_var,
            momentum,
            eps,
            count_all.view(-1)
        )
        dummy_mean = mean.new_zeros(size=mean.size(), dtype=mean.dtype, device=mean.device)
        mean = dummy_mean

        self.save_for_backward(input, weight, mean, invstd, count_all.to(torch.int32))
        self.process_group = process_group

        # apply element-wise normalization
        if input.numel() > 0:
            return torch.batch_norm_elemt(input, weight, bias, mean, invstd, eps)
        else:
            return torch.empty_like(input)

    @staticmethod
    def backward(self, grad_output):
        if not (
            grad_output.is_contiguous(memory_format=torch.channels_last) or
            grad_output.is_contiguous(memory_format=torch.channels_last_3d)
        ):
            grad_output = grad_output.contiguous()
        saved_input, weight, mean, invstd, count_tensor = self.saved_tensors
        grad_input = grad_weight = grad_bias = None
        process_group = self.process_group

        if saved_input.numel() > 0:
            # calculate local stats as well as grad_weight / grad_bias
            sum_dy, sum_dy_xmu, grad_weight, grad_bias = torch.batch_norm_backward_reduce(
                grad_output,
                saved_input,
                mean,
                invstd,
                weight,
                self.needs_input_grad[0],
                self.needs_input_grad[1],
                self.needs_input_grad[2]
            )

            if self.needs_input_grad[0]:
                # synchronizing stats used to calculate input gradient.
                num_channels = sum_dy.shape[0]
                combined = torch.cat([sum_dy, sum_dy_xmu], dim=0)
                torch.distributed.all_reduce(
                    combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)
                sum_dy, sum_dy_xmu = torch.split(combined, num_channels)

                # backward pass for gradient calculation
                grad_input = torch.batch_norm_backward_elemt(
                    grad_output,
                    saved_input,
                    mean,
                    invstd,
                    weight,
                    sum_dy,
                    sum_dy_xmu,
                    count_tensor
                )
            # synchronizing of grad_weight / grad_bias is not needed as distributed
            # training would handle all reduce.
            if weight is None or not self.needs_input_grad[1]:
                grad_weight = None

            if weight is None or not self.needs_input_grad[2]:
                grad_bias = None
        else:
            # This process got an empty input tensor in the forward pass.
            # Although this process can directly set grad_input as an empty
            # tensor of zeros, it still needs to participate in the collective
            # communication to unblock its peers, as other peer processes might
            # have recieved non-empty inputs.
            num_channels = saved_input.shape[1]
            if self.needs_input_grad[0]:
                # launch all_reduce to unblock other peer processes
                combined = torch.zeros(
                    2 * num_channels,
                    dtype=saved_input.dtype,
                    device=saved_input.device
                )
                torch.distributed.all_reduce(
                    combined, torch.distributed.ReduceOp.SUM, process_group, async_op=False)

            # Leave grad_input, grad_weight and grad_bias as None, which will be
            # interpreted by the autograd engine as Tensors full of zeros.

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None

