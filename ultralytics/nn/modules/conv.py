# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Convolution modules."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

__all__ = (
    "DSC",
    "SimAM",
    "SimConv",
    "CBAM",
    "ChannelAttention",
    "Concat",
    "Conv",
    "Conv2",
    "ConvTranspose",
    "DWConv",
    "DWConvTranspose2d",
    "Focus",
    "GhostConv",
    "Index",
    "LightConv",
    "RepConv",
    "SpatialAttention",
    "BiFormer"
)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution module with batch normalization and activation.

    Attributes:
        conv (nn.Conv2d): Convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.Mish()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Apply convolution and activation without batch normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))


class Conv2(Conv):
    """Simplified RepConv module with Conv fusing.

    Attributes:
        conv (nn.Conv2d): Main 3x3 convolutional layer.
        cv2 (nn.Conv2d): Additional 1x1 convolutional layer.
        bn (nn.BatchNorm2d): Batch normalization layer.
        act (nn.Module): Activation function layer.
    """

    def __init__(self, c1, c2, k=3, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv2 layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, p, g=g, d=d, act=act)
        self.cv2 = nn.Conv2d(c1, c2, 1, s, autopad(1, p, d), groups=g, dilation=d, bias=False)  # add 1x1 conv

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x) + self.cv2(x)))

    def forward_fuse(self, x):
        """Apply fused convolution, batch normalization and activation to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv(x)))

    def fuse_convs(self):
        """Fuse parallel convolutions."""
        w = torch.zeros_like(self.conv.weight.data)
        i = [x // 2 for x in w.shape[2:]]
        w[:, :, i[0] : i[0] + 1, i[1] : i[1] + 1] = self.cv2.weight.data.clone()
        self.conv.weight.data += w
        self.__delattr__("cv2")
        self.forward = self.forward_fuse


class LightConv(nn.Module):
    """Light convolution module with 1x1 and depthwise convolutions.

    This implementation is based on the PaddleDetection HGNetV2 backbone.

    Attributes:
        conv1 (Conv): 1x1 convolution layer.
        conv2 (DWConv): Depthwise convolution layer.
    """

    def __init__(self, c1, c2, k=1, act=nn.ReLU()):
        """Initialize LightConv layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size for depthwise convolution.
            act (nn.Module): Activation function.
        """
        super().__init__()
        self.conv1 = Conv(c1, c2, 1, act=False)
        self.conv2 = DWConv(c2, c2, k, act=act)

    def forward(self, x):
        """Apply 2 convolutions to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv2(self.conv1(x))


class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    """Depth-wise transpose convolution module."""

    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):
        """Initialize depth-wise transpose convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p1 (int): Padding.
            p2 (int): Output padding.
        """
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class ConvTranspose(nn.Module):
    """Convolution transpose module with optional batch normalization and activation.

    Attributes:
        conv_transpose (nn.ConvTranspose2d): Transposed convolution layer.
        bn (nn.BatchNorm2d | nn.Identity): Batch normalization layer.
        act (nn.Module): Activation function layer.
        default_act (nn.Module): Default activation function (SiLU).
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=2, s=2, p=0, bn=True, act=True):
        """Initialize ConvTranspose layer with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            bn (bool): Use batch normalization.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(c1, c2, k, s, p, bias=not bn)
        self.bn = nn.BatchNorm2d(c2) if bn else nn.Identity()
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply transposed convolution, batch normalization and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.bn(self.conv_transpose(x)))

    def forward_fuse(self, x):
        """Apply convolution transpose and activation to input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv_transpose(x))


class Focus(nn.Module):
    """Focus module for concentrating feature information.

    Slices input tensor into 4 parts and concatenates them in the channel dimension.

    Attributes:
        conv (Conv): Convolution layer.
    """

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        """Initialize Focus module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int, optional): Padding.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):
        """Apply Focus operation and convolution to input tensor.

        Input shape is (B, C, H, W) and output shape is (B, c2, H/2, W/2).

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    """Ghost Convolution module.

    Generates more features with fewer parameters by using cheap operations.

    Attributes:
        cv1 (Conv): Primary convolution.
        cv2 (Conv): Cheap operation convolution.

    References:
        https://github.com/huawei-noah/Efficient-AI-Backbones
    """

    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        """Initialize Ghost Convolution module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            g (int): Groups.
            act (bool | nn.Module): Activation function.
        """
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        """Apply Ghost Convolution to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor with concatenated features.
        """
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class RepConv(nn.Module):
    """RepConv module with training and deploy modes.

    This module is used in RT-DETR and can fuse convolutions during inference for efficiency.

    Attributes:
        conv1 (Conv): 3x3 convolution.
        conv2 (Conv): 1x1 convolution.
        bn (nn.BatchNorm2d, optional): Batch normalization for identity branch.
        act (nn.Module): Activation function.
        default_act (nn.Module): Default activation function (SiLU).

    References:
        https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        """Initialize RepConv module with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            p (int): Padding.
            g (int): Groups.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
            bn (bool): Use batch normalization for identity branch.
            deploy (bool): Deploy mode for inference.
        """
        super().__init__()
        assert k == 3 and p == 1
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.bn = nn.BatchNorm2d(num_features=c1) if bn and c2 == c1 and s == 1 else None
        self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
        self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward_fuse(self, x):
        """Forward pass for deploy mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return self.act(self.conv(x))

    def forward(self, x):
        """Forward pass for training mode.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        """Calculate equivalent kernel and bias by fusing convolutions.

        Returns:
            (torch.Tensor): Equivalent kernel
            (torch.Tensor): Equivalent bias
        """
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    @staticmethod
    def _pad_1x1_to_3x3_tensor(kernel1x1):
        """Pad a 1x1 kernel to 3x3 size.

        Args:
            kernel1x1 (torch.Tensor): 1x1 convolution kernel.

        Returns:
            (torch.Tensor): Padded 3x3 kernel.
        """
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        """Fuse batch normalization with convolution weights.

        Args:
            branch (Conv | nn.BatchNorm2d | None): Branch to fuse.

        Returns:
            kernel (torch.Tensor): Fused kernel.
            bias (torch.Tensor): Fused bias.
        """
        if branch is None:
            return 0, 0
        if isinstance(branch, Conv):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = np.zeros((self.c1, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def fuse_convs(self):
        """Fuse convolutions for inference by creating a single equivalent convolution."""
        if hasattr(self, "conv"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(
            in_channels=self.conv1.conv.in_channels,
            out_channels=self.conv1.conv.out_channels,
            kernel_size=self.conv1.conv.kernel_size,
            stride=self.conv1.conv.stride,
            padding=self.conv1.conv.padding,
            dilation=self.conv1.conv.dilation,
            groups=self.conv1.conv.groups,
            bias=True,
        ).requires_grad_(False)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("conv1")
        self.__delattr__("conv2")
        if hasattr(self, "nm"):
            self.__delattr__("nm")
        if hasattr(self, "bn"):
            self.__delattr__("bn")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")


class ChannelAttention(nn.Module):
    """Channel-attention module for feature recalibration.

    Applies attention weights to channels based on global average pooling.

    Attributes:
        pool (nn.AdaptiveAvgPool2d): Global average pooling.
        fc (nn.Conv2d): Fully connected layer implemented as 1x1 convolution.
        act (nn.Sigmoid): Sigmoid activation for attention weights.

    References:
        https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet
    """

    def __init__(self, channels: int) -> None:
        """Initialize Channel-attention module.

        Args:
            channels (int): Number of input channels.
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Channel-attended output tensor.
        """
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module for feature recalibration.

    Applies attention weights to spatial dimensions based on channel statistics.

    Attributes:
        cv1 (nn.Conv2d): Convolution layer for spatial attention.
        act (nn.Sigmoid): Sigmoid activation for attention weights.
    """

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module.

        Args:
            kernel_size (int): Size of the convolutional kernel (3 or 7).
        """
        super().__init__()
        assert kernel_size in {3, 7}, "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply spatial attention to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Spatial-attended output tensor.
        """
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module.

    Combines channel and spatial attention mechanisms for comprehensive feature refinement.

    Attributes:
        channel_attention (ChannelAttention): Channel attention module.
        spatial_attention (SpatialAttention): Spatial attention module.
    """

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given parameters.

        Args:
            c1 (int): Number of input channels.
            kernel_size (int): Size of the convolutional kernel for spatial attention.
        """
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Apply channel and spatial attention sequentially to input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Attended output tensor.
        """
        return self.spatial_attention(self.channel_attention(x))


class Concat(nn.Module):
    """Concatenate a list of tensors along specified dimension.

    Attributes:
        d (int): Dimension along which to concatenate tensors.
    """

    def __init__(self, dimension=1):
        """Initialize Concat module.

        Args:
            dimension (int): Dimension along which to concatenate tensors.
        """
        super().__init__()
        self.d = dimension

    def forward(self, x: list[torch.Tensor]):
        """Concatenate input tensors along specified dimension.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Concatenated tensor.
        """
        return torch.cat(x, self.d)


class Index(nn.Module):
    """Returns a particular index of the input.

    Attributes:
        index (int): Index to select from input.
    """

    def __init__(self, index=0):
        """Initialize Index module.

        Args:
            index (int): Index to select from input.
        """
        super().__init__()
        self.index = index

    def forward(self, x: list[torch.Tensor]):
        """Select and return a particular index from input.

        Args:
            x (list[torch.Tensor]): List of input tensors.

        Returns:
            (torch.Tensor): Selected tensor.
        """
        return x[self.index]

# Fix class for paper:
class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "lambda=%f)" % self.e_lambda
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = (
            x_minus_mu_square
            / (
                4
                * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)
            )
            + 0.5
        )

        return x * self.activaton(y)


class SimConv(nn.Module):
    """Normal Conv with ReLU VAN_activation"""

    def __init__(self, c1, c2, k=1, s=1, g=1, d=1, bias=False, p=None):
        super().__init__()

        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        # self.act = nn.LeakyReLU()
        self.act = nn.Mish()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# Yolo-gold paper:
import torch.nn.functional as F


def conv_bn(
    in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False
):
    """Basic cell for rep-style block, including conv and bn"""
    result = nn.Sequential()
    result.add_module(
        "conv",
        nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias,
        ),
    )
    result.add_module("bn", nn.BatchNorm2d(num_features=out_channels))
    return result


class RepVGGBlock(nn.Module):
    """RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        padding_mode="zeros",
        deploy=False,
        use_se=False,
    ):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        assert kernel_size == 3
        assert padding == 1

        padding_11 = padding - kernel_size // 2

        self.nonlinearity = nn.ReLU()

        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=True,
                padding_mode=padding_mode,
            )

        else:
            self.rbr_identity = (
                nn.BatchNorm2d(num_features=in_channels)
                if out_channels == in_channels and stride == 1
                else None
            )
            self.rbr_dense = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
            )
            self.rbr_1x1 = conv_bn(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                padding=padding_11,
                groups=groups,
            )

    def forward(self, inputs):
        """Forward process"""
        if hasattr(self, "rbr_reparam"):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        return self.nonlinearity(
            self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out)
        )

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return (
            kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid,
            bias3x3 + bias1x1 + biasid,
        )

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros(
                    (self.in_channels, input_dim, 3, 3), dtype=np.float32
                )
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

    def switch_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
            in_channels=self.rbr_dense.conv.in_channels,
            out_channels=self.rbr_dense.conv.out_channels,
            kernel_size=self.rbr_dense.conv.kernel_size,
            stride=self.rbr_dense.conv.stride,
            padding=self.rbr_dense.conv.padding,
            dilation=self.rbr_dense.conv.dilation,
            groups=self.rbr_dense.conv.groups,
            bias=True,
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True


def onnx_AdaptiveAvgPool2d(x, output_size):
    stride_size = np.floor(np.array(x.shape[-2:]) / output_size).astype(np.int32)
    kernel_size = np.array(x.shape[-2:]) - (output_size - 1) * stride_size
    avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
    x = avg(x)
    return x


def get_avg_pool():
    if torch.onnx.is_in_onnx_export():
        avg_pool = onnx_AdaptiveAvgPool2d
    else:
        avg_pool = nn.functional.adaptive_avg_pool2d
    return avg_pool


class SimFusion_3in(nn.Module):
    def __init__(self, in_channel_list, out_channels):
        super().__init__()
        self.cv1 = (
            Conv(in_channel_list[0], out_channels, act=nn.ReLU())
            if in_channel_list[0] != out_channels
            else nn.Identity()
        )
        self.cv2 = (
            Conv(in_channel_list[1], out_channels, act=nn.ReLU())
            if in_channel_list[1] != out_channels
            else nn.Identity()
        )
        self.cv3 = (
            Conv(in_channel_list[2], out_channels, act=nn.ReLU())
            if in_channel_list[2] != out_channels
            else nn.Identity()
        )
        self.cv_fuse = Conv(out_channels * 3, out_channels, act=nn.ReLU())
        self.downsample = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        N, C, H, W = x[1].shape
        output_size = (H, W)

        if torch.onnx.is_in_onnx_export():
            self.downsample = onnx_AdaptiveAvgPool2d
            output_size = np.array([H, W])

        x0 = self.cv1(self.downsample(x[0], output_size))
        x1 = self.cv2(x[1])
        x2 = self.cv3(
            F.interpolate(x[2], size=(H, W), mode="bilinear", align_corners=False)
        )
        return self.cv_fuse(torch.cat((x0, x1, x2), dim=1))


class SimFusion_4in(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.functional.adaptive_avg_pool2d

    def forward(self, x):
        x_l, x_m, x_s, x_n = x
        B, C, H, W = x_s.shape
        output_size = np.array([H, W])

        if torch.onnx.is_in_onnx_export():
            self.avg_pool = onnx_AdaptiveAvgPool2d

        x_l = self.avg_pool(x_l, output_size)
        x_m = self.avg_pool(x_m, output_size)
        x_n = F.interpolate(x_n, size=(H, W), mode="bilinear", align_corners=False)

        out = torch.cat([x_l, x_m, x_s, x_n], 1)
        return out

class IFM(nn.Module):
    def __init__(self, inc, ouc, embed_dim_p=96, fuse_block_num=3) -> None:
        super().__init__()

        self.conv = nn.Sequential(
            Conv(inc, embed_dim_p),
            *[RepVGGBlock(embed_dim_p, embed_dim_p) for _ in range(fuse_block_num)],
            Conv(embed_dim_p, sum(ouc))
        )

    def forward(self, x):
        return self.conv(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class InjectionMultiSum_Auto_pool(nn.Module):
    def __init__(self, inp: int, oup: int, global_inp: list, flag: int) -> None:
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        self.local_embedding = Conv(inp, oup, 1, act=False)
        self.global_embedding = Conv(global_inp[self.flag], oup, 1, act=False)
        self.global_act = Conv(global_inp[self.flag], oup, 1, act=False)
        self.act = h_sigmoid()

    def forward(self, x):
        """
        x_g: global features
        x_l: local features
        """
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]

        local_feat = self.local_embedding(x_l)

        global_act = self.global_act(gloabl_info)
        global_feat = self.global_embedding(gloabl_info)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])

            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)

        else:
            sig_act = F.interpolate(
                self.act(global_act), size=(H, W), mode="bilinear", align_corners=False
            )
            global_feat = F.interpolate(
                global_feat, size=(H, W), mode="bilinear", align_corners=False
            )

        out = local_feat * sig_act + global_feat
        return out


def get_shape(tensor):
    shape = tensor.shape
    if torch.onnx.is_in_onnx_export():
        shape = [i.cpu().numpy() for i in shape]
    return shape


class PyramidPoolAgg(nn.Module):
    def __init__(self, inc, ouc, stride, pool_mode="torch"):
        super().__init__()
        self.stride = stride
        if pool_mode == "torch":
            self.pool = nn.functional.adaptive_avg_pool2d
        elif pool_mode == "onnx":
            self.pool = onnx_AdaptiveAvgPool2d
        self.conv = Conv(inc, ouc)

    def forward(self, inputs):
        B, C, H, W = get_shape(inputs[-1])
        H = (H - 1) // self.stride + 1
        W = (W - 1) // self.stride + 1

        output_size = np.array([H, W])

        if not hasattr(self, "pool"):
            self.pool = nn.functional.adaptive_avg_pool2d

        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d

        out = [self.pool(inp, output_size) for inp in inputs]

        return self.conv(torch.cat(out, dim=1))


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (
        x.ndim - 1
    )  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = Conv(in_features, hidden_features, act=False)
        self.dwconv = nn.Conv2d(
            hidden_features, hidden_features, 3, 1, 1, bias=True, groups=hidden_features
        )
        self.act = nn.ReLU6()
        self.fc2 = Conv(hidden_features, out_features, act=False)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads, attn_ratio=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim**-0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = Conv(dim, nh_kd, 1, act=False)
        self.to_k = Conv(dim, nh_kd, 1, act=False)
        self.to_v = Conv(dim, self.dh, 1, act=False)

        self.proj = torch.nn.Sequential(nn.ReLU6(), Conv(self.dh, dim, act=False))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = get_shape(x)

        qq = (
            self.to_q(x)
            .reshape(B, self.num_heads, self.key_dim, H * W)
            .permute(0, 1, 3, 2)
        )
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class top_Block(nn.Module):

    def __init__(
        self,
        dim,
        key_dim,
        num_heads,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.attn = Attention(
            dim, key_dim=key_dim, num_heads=num_heads, attn_ratio=attn_ratio
        )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x1):
        x1 = x1 + self.drop_path(self.attn(x1))
        x1 = x1 + self.drop_path(self.mlp(x1))
        return x1


class TopBasicLayer(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ouc_list,
        block_num=2,
        key_dim=8,
        num_heads=4,
        mlp_ratio=4.0,
        attn_ratio=2.0,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
    ):
        super().__init__()
        self.block_num = block_num

        self.transformer_blocks = nn.ModuleList()
        for i in range(self.block_num):
            self.transformer_blocks.append(
                top_Block(
                    embedding_dim,
                    key_dim=key_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_ratio=attn_ratio,
                    drop=drop,
                    drop_path=(
                        drop_path[i] if isinstance(drop_path, list) else drop_path
                    ),
                )
            )
        self.conv = nn.Conv2d(embedding_dim, sum(ouc_list), 1)

    def forward(self, x):
        # token * N
        for i in range(self.block_num):
            x = self.transformer_blocks[i](x)
        return self.conv(x)


class AdvPoolFusion(nn.Module):
    def forward(self, x):
        x1, x2 = x
        if torch.onnx.is_in_onnx_export():
            self.pool = onnx_AdaptiveAvgPool2d
        else:
            self.pool = nn.functional.adaptive_avg_pool2d

        N, C, H, W = x2.shape
        output_size = np.array([H, W])
        x1 = self.pool(x1, output_size)

        return torch.cat([x1, x2], 1)

class DSC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True):
        super(DSC, self).__init__()
        
        # Hàm kích hoạt mặc định của YOLOv8 là SiLU (Swish)
        # Nếu act=False thì dùng nn.Identity() (không làm gì cả, giữ nguyên đặc trưng)
        activation = nn.SiLU() if act else nn.Identity()
        
        # BƯỚC 1: DEPTHWISE CONVOLUTION (DWConv)
        # Điểm mấu chốt ở đây là tham số: groups = in_channels
        # Nó bắt buộc PyTorch phải tách riêng từng channel ra để tính toán độc lập
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=in_channels, # Số lượng kênh ra bằng số lượng kênh vào
                kernel_size=kernel_size, 
                stride=stride, 
                padding=padding, 
                groups=in_channels,       # <--- ĐÂY LÀ CHÌA KHÓA TOÁN HỌC!
                bias=False
            ),
            nn.BatchNorm2d(in_channels),
            activation
        )
        
        # BƯỚC 2: POINTWISE CONVOLUTION (PWConv)
        # Sử dụng kernel kích thước cố định là 1x1 để trộn thông tin xuyên kênh
        # và thay đổi số lượng kênh đầu ra theo ý muốn (out_channels)
        self.pointwise = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=1, 
                stride=1, 
                padding=0, 
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            activation
        )

    def forward(self, x):
        # Luồng dữ liệu: Input -> DWConv -> PWConv -> Output
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
    
class PConv1x1(nn.Module):
    def __init__(self, inp, oup, n_div=4):
        super().__init__()
        # Chia làm 2 nhánh theo tỷ lệ n_div
        self.dim_conv = max(1, inp // n_div) # Đảm bảo ít nhất có 1 kênh đem đi conv
        self.dim_untouched = inp - self.dim_conv
        
        # Tính toán số kênh đầu ra cho 2 nhánh sao cho tổng bằng oup
        # Nhánh 1 (tích chập) sẽ gánh một nửa oup, nhánh 2 (giữ nguyên/linear) gánh nửa còn lại
        self.oup_conv = max(1, oup // n_div)
        self.oup_untouched = oup - self.oup_conv
        
        # Nhánh 1: Tích chập thay đổi số kênh
        self.partial_conv = nn.Conv2d(self.dim_conv, self.oup_conv, kernel_size=1, bias=False)
        
        # Nhánh 2: Nếu số kênh đầu vào và đầu ra của nhánh 2 khác nhau, 
        # dùng một lớp Conv 1x1 siêu nhẹ (hoặc AvgPool) để điều chỉnh, không để giữ nguyên thuần túy
        if self.dim_untouched != self.oup_untouched:
            self.identity_adjust = nn.Conv2d(self.dim_untouched, self.oup_untouched, kernel_size=1, bias=False)
        else:
            self.identity_adjust = nn.Identity()

    def forward(self, x):
        # Chia tách kênh theo đầu vào
        x1, x2 = torch.split(x, [self.dim_conv, self.dim_untouched], dim=1)
        
        # Xử lý nhánh 1 (Tích chập)
        x1 = self.partial_conv(x1)
        
        # Xử lý nhánh 2 (Điều chỉnh số kênh cho khớp đầu ra)
        x2 = self.identity_adjust(x2)
        
        # Ghép lại với nhau -> Tổng số kênh chắc chắn bằng oup
        return torch.cat((x1, x2), dim=1)
class Inject_PConv(nn.Module):
    def __init__(self, inp: int, oup: int, global_inp: list, flag: int) -> None:
        super().__init__()
        self.global_inp = global_inp
        self.flag = flag
        g_inp = global_inp[self.flag]
        
        # THAY THẾ: Sử dụng PConv thay cho Conv 1x1 truyền thống giúp nén mạng
        self.local_embedding = PConv1x1(inp, oup)
        self.global_embedding = PConv1x1(g_inp, oup)
        self.global_act = PConv1x1(g_inp, oup)
        
        self.act = h_sigmoid() # Giữ nguyên hàm kích hoạt của Gold-YOLO

    def forward(self, x):
        x_l, x_g = x
        B, C, H, W = x_l.shape
        g_B, g_C, g_H, g_W = x_g.shape
        use_pool = H < g_H

        gloabl_info = x_g.split(self.global_inp, dim=1)[self.flag]

        # Khối PConv xử lý
        local_feat = self.local_embedding(x_l)
        global_act = self.global_act(gloabl_info)
        global_feat = self.global_embedding(gloabl_info)

        if use_pool:
            avg_pool = get_avg_pool()
            output_size = np.array([H, W])
            sig_act = avg_pool(global_act, output_size)
            global_feat = avg_pool(global_feat, output_size)
        else:
            sig_act = F.interpolate(
                self.act(global_act), size=(H, W), mode="bilinear", align_corners=False
            )
            global_feat = F.interpolate(
                global_feat, size=(H, W), mode="bilinear", align_corners=False
            )

        # Trộn đặc trưng 2 ngõ vào (Giữ nguyên logic của Gold-YOLO)
        out = local_feat * sig_act + global_feat
        return out
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLevelRoutingAttention(nn.Module):
    """
    Bi-level Routing Attention (BRA) - Đã sửa lỗi định tuyến thu thập mã Token bằng Torch.Gather
    """
    def __init__(self, dim, num_heads=8, n_win=7, topk=4, qkv_bias=True, qk_scale=None, side_dwconv=3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.n_win = n_win
        self.topk = topk
        self.scale = qk_scale or self.head_dim ** -0.5

        # Gom bộ chiếu QKV thành 1 tầng Linear tuyến tính tăng tốc độ xử lý
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.output_linear = nn.Linear(dim, dim)
        self.router_linear = nn.Linear(dim, dim)

        if side_dwconv > 0:
            self.dwconv = nn.Conv2d(dim, dim, kernel_size=side_dwconv, 
                                    padding=side_dwconv // 2, groups=dim)
        else:
            self.dwconv = None

    def forward(self, x):
        """
        Input: x có kích thước [B, H, W, C]
        """
        B, H, W, C = x.shape
        S = self.n_win
        
        # 1. TÍNH TOÁN PADDING ĐỘNG ĐỂ CHIA HẾT CHO S (CỨU LỖI RUNTIME SHAPE)
        pad_h = (S - H % S) % S
        pad_w = (S - W % S) % S
        
        if pad_h > 0 or pad_w > 0:
            # Thực hiện thêm viền đệm (Padding) zero vào phía sau các chiều H và W
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h)) # Định dạng pad: (C_trước, C_sau, W_trước, W_sau, H_trước, H_sau)
            _, H_pad, W_pad, _ = x.shape
        else:
            H_pad, W_pad = H, W
            
        window_size_h = H_pad // S
        window_size_w = W_pad // S
        n_tokens_per_win = window_size_h * window_size_w
        
        # Nhánh tăng cường vị trí cục bộ bằng DWConv
        if self.dwconv is not None:
            x_conv = x.permute(0, 3, 1, 2).contiguous()
            x_conv = self.dwconv(x_conv)
            x_conv = x_conv.permute(0, 2, 3, 1).contiguous()
            x = x + x_conv

        # Chiếu ma trận tách biệt Q, K, V
        qkv = self.qkv(x) 
        q, k, v = qkv.chunk(3, dim=-1) 

        # Phân chia cấu trúc không gian thành các ô cửa sổ độc lập (Window Partition)
        # Bấy giờ H_pad và W_pad chắc chắn chia hết cho S (n_win) nên không bao giờ lỗi nữa!
        q_win = q.view(B, S, window_size_h, S, window_size_w, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, S*S, n_tokens_per_win, C)
        k_win = k.view(B, S, window_size_h, S, window_size_w, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, S*S, n_tokens_per_win, C)
        v_win = v.view(B, S, window_size_h, S, window_size_w, C).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, S*S, n_tokens_per_win, C)
        
        # 2. ĐỊNH TUYẾN CẤP ĐỘ VÙNG (Region-level routing)
        q_r = self.router_linear(q_win.mean(dim=2)) 
        k_r = self.router_linear(k_win.mean(dim=2)) 
        
        router_attn = (q_r @ k_r.transpose(-2, -1)) * self.scale
        router_attn = router_attn.softmax(dim=-1) 
        
        topk_attn, topk_indices = torch.topk(router_attn, k=self.topk, dim=-1) 
        
        # 3. THU THẬP TOKEN TỪ CÁC VÙNG ĐƯỢC CHỈ ĐỊNH
        gather_indices = topk_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, n_tokens_per_win, C)
        
        k_win_expanded = k_win.unsqueeze(1).expand(-1, S*S, -1, -1, -1)
        v_win_expanded = v_win.unsqueeze(1).expand(-1, S*S, -1, -1, -1)
        
        k_routed = torch.gather(k_win_expanded, dim=2, index=gather_indices).flatten(2, 3) 
        v_routed = torch.gather(v_win_expanded, dim=2, index=gather_indices).flatten(2, 3) 

        # 4. TÍNH TOÁN ĐA ĐẦU CHÚ Ý CHI TIẾT (Token-to-Token Attention)
        q_win = q_win.view(B, S*S, n_tokens_per_win, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k_routed = k_routed.view(B, S*S, self.topk * n_tokens_per_win, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)
        v_routed = v_routed.view(B, S*S, self.topk * n_tokens_per_win, self.num_heads, self.head_dim).permute(0, 1, 3, 2, 4)

        attn = (q_win @ k_routed.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v_routed).permute(0, 1, 3, 2, 4).reshape(B, S*S, n_tokens_per_win, C)
        
        # 5. KHÔI PHỤC ĐỊNH DẠNG KHÔNG GIAN VÀ CẮT BỎ PADDING THỪA ĐỂ TRẢ LẠI ĐÚNG SHAPE GỐC
        out = out.view(B, S, S, window_size_h, window_size_w, C).permute(0, 1, 3, 2, 4, 5).reshape(B, H_pad, W_pad, C)
        
        if pad_h > 0 or pad_w > 0:
            out = out[:, :H, :W, :].contiguous() # Cắt bỏ phần rìa ảnh ảo đã đệm ban đầu
            
        out = self.output_linear(out)
        return out

# ==========================================
# 2. KHỐI BIFOMERBLOCK HOÀN CHỈNH
# ==========================================
class BiFormerBlock(nn.Module):
    def __init__(self, dim, num_heads, n_win=7, topk=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., drop_path=0., norm_layer=nn.LayerNorm, side_dwconv=3):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = BiLevelRoutingAttention(
            dim=dim, num_heads=num_heads, n_win=n_win, topk=topk,
            qkv_bias=qkv_bias, qk_scale=qk_scale, side_dwconv=side_dwconv
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Ml_p(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

    def forward(self, x):
        is_channels_first = False
        if len(x.shape) == 4 and x.shape[1] != x.shape[2] and x.shape[1] != x.shape[3]:
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous()
            is_channels_first = True

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if is_channels_first:
            x = x.permute(0, 3, 1, 2).contiguous()
        return x

# ==========================================
# 3. MODULE KẾT HỢP DÀNH CHO YOLO (YAML Wrapper)
# ==========================================
class BiFormer(nn.Module):
    def __init__(self, c1, c2, num_heads=8, n_win=7, topk=4, mlp_ratio=4.):
        super().__init__()
        self.conv_match = nn.Conv2d(c1, c2, kernel_size=1) if c1 != c2 else nn.Identity()
        self.block = BiFormerBlock(dim=c2, num_heads=num_heads, n_win=n_win, topk=topk, mlp_ratio=mlp_ratio)

    def forward(self, x):
        x = self.conv_match(x)
        return self.block(x)
    
class Ml_p(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))