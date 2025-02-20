from collections import OrderedDict
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

__all__ = ["FourierNet2D"]

### UTILS
def _list(x, repetitions=1):
    if hasattr(x, "__iter__") and not isinstance(x, str):
        return x
    else:
        return [
            x,
        ] * repetitions

class InputScalingSequential(nn.Sequential):
    def __init__(self, quantile, quantile_scale, *args):
        super().__init__(*args)
        self.quantile = quantile
        self.quantile_scale = quantile_scale
        
    def forward(self, input):
        # unscale input by quantile and record quantile scale
        quantile = torch.quantile(input, self.quantile_scale).detach()
        scale = self.quantile_scale * quantile
        input = input / scale
        for module in self:
            input = module(input)
        input = input * scale
        return input
    
    
class InputNormingSequential(nn.Sequential):
    # mean zero and unit variance.
    def __init__(self, dim, *args):
        super().__init__(*args)
        self.dim = dim

    def forward(self, input):
        if self.dim is None:
            shift = torch.mean(input).detach()
        else:
            shift = torch.mean(input, self.dim).detach()
        input = input - shift.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        # scale sample to unit variance and record scale
        if self.dim is None:
            scale = torch.sqrt(torch.var(input).detach() + 1e-5)
        else:
            scale = torch.sqrt(
                torch.var(input, self.dim, unbiased=False).detach() + 1e-5
            )
        input = input / scale.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        for module in self:
            input = module(input)
        # rescale output by recorded scale
        input = input * scale.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        # reshift output by recorded shift
        input = input + shift.view(
            tuple(1 if d != 1 else -1 for d in range(len(input.shape)))
        )
        return input


class FourierConv2D(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size, ### This kernel size should be same with input size.
        stride=1,
        padding=True,
        bias=True,
        reduce_channels=True,
    ):
        super(FourierConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        kernel_size = nn.modules.utils._pair(kernel_size)
        self.stride = nn.modules.utils._pair(stride)
        self.padding = padding
        
        if self.padding and self.stride == (1, 1):
            self.kernel_size = (kernel_size[0]*2 - 1, kernel_size[1] * 2 - 1)
        else:
            self.kernel_size = kernel_size
        self.weight = nn.Parameter(
            torch.Tensor(out_channels, in_channels, *self.kernel_size, dtype=torch.cfloat)
        )
        self.reduce_channels = reduce_channels
        if bias and self.reduce_channels:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1, 1))
        elif bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels, 1, 1, 1))
        else:
            self.register_parameter("bias", 0)
        self.reset_parameters()
        
    def reset_parameters(self):
        init.kaiming_normal_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)
            
    def extra_repr(self):
        s = "{in_channels}, {out_channels}, kernel_size={kernel_size}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)
    
    def forward(self, im):
        # im : B, C, H, W
        insize = im.size()
        outsize = (self.out_channels, im.size(-2), im.size(-1))
        fftsize = [insize[-2] + outsize[-2] -1, insize[-1] + outsize[-1] - 1]
        
        if self.padding:
            pad_size = (
                0, fftsize[-1] - insize[-1], 0, fftsize[-2] - insize[-2]
            )
        else:
            pad_size = ()
        
        padded_im = F.pad(im, pad_size)
        rfft_out_size = padded_im.shape
        fourier_im = torch.fft.rfft2(padded_im, norm='ortho') # B, C, H', W'
        # weight : C', C, K, K
        
        # TODO:: Understanding the stride operation..
        if self.stride != (1,1):
            stride = self.stride
            fourier_im = fourier_im[:, :, :: stride[-2], :: stride[-1]]
        ##### fourier_feats' shape and weight's shape should be same.!! (spatial shape.)
        fourier_feats = fourier_im.unsqueeze(1) * self.weight
        
        real_feats = torch.fft.rfft2(fourier_feats, norm='orth', s=(rfft_out_size[-2:]))
        # B, C', C, H', W'
        
        if self.padding and self.stride == (1, 1):
            cropsize = [fftsize[-2] - insize[-1], fftsize[-1] - insize[-1]]
            cropsize_left = [int(c/2) for c in cropsize]
            cropsize_right = [int((c+1)/2) for c in cropsize]
            real_feats = F.pad(
                real_feats,
                (
                    -cropsize_left[-1],
                    -cropsize_right[-1],
                    -cropsize_left[-2],
                    -cropsize_right[-2]
                )
            )
        if self.reduce_channels:
            real_feats = real_feats.sum(2) # B, C', H', W'
                
        real_feats = real_feats + self.bias
            
        return real_feats
    
    
def FourierNet2D(
    fourier_out,
    fourier_kernel_size,
    num_planes, # z axis num.
    fourier_conv_args=None,
    conv_kernel_size=[11],
    conv_fmap_nums=None,
    input_scaling_mode="batchnorm",
    quantile=0.5,
    quantile_scale=1.0,
    imbn_momentum=0.1,
    fourierbn_momentum=0.1,
    convbn_momentum=0.1,
    device='cpu'
):
    
    conv_kernel_sizes = _list(conv_kernel_sizes)
    # conv_fmap_nums : convolutional layers' channels
    if conv_fmap_nums is None:
        conv_fmap_nums = [num_planes] * len(conv_kernel_sizes)
    assert conv_fmap_nums[-1] == num_planes, "Must output number of planes"
    
    if input_scaling_mode in ["scaling", "norming"]:
        layers = []
    elif input_scaling_mode in ["batchnorm"]:
        imbn = nn.BatchNorm2d(1, momentum=imbn_momentum,).to(device)
        layers = [("image_bn", imbn)]
    else:
        raise TypeError(
            "Argument input_scaling_mode must be one of 'scaling', 'norming', or 'batchnorm'"
        )
    if fourier_conv_args is None:
        fourier_conv_args = {}
    
    fourier_conv = FourierConv2D(
        1, fourier_out, fourier_kernel_size, **fourier_conv_args
    ).to(device)
    fourier_relu = nn.LeakyReLU().to(device)
    fourierbn = nn.BatchNorm2d(fourier_out, momentum=fourierbn_momentum).to(device)
    layers += [
        ("fourier_conv", fourier_conv),
        ("fourier_relu", fourier_relu),
        ("fourier_bn", fourierbn)
    ]
    
    #### After a Fourier layer (global information), apply additional convolutional layers.
    
    for i in range(len(conv_fmap_nums)):
        previous_fmap_nums = fourier_out if i == 0 else conv_fmap_nums[i-1]
        conv = nn.Conv2d(
            previous_fmap_nums,
            conv_fmap_nums[i],
            conv_kernel_sizes[i],
            padding=int(math.floor(conv_kernel_sizes[i]/2))
        )
        conv = conv.to(device)
        layers.append((f"conv2d_{i+1}", conv))
        if i < len(conv_fmap_nums) -1:
            conv_relu = nn.LeakyReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
            conv_bn = nn.BatchNorm2d(conv_fmap_nums[i], momentum=convbn_momentum).to(
                device
            )
            layers.append((f"conv{i+1}_bn", conv_bn))
        else:
            conv_relu = nn.ReLU().to(device)
            layers.append((f"conv{i+1}_relu", conv_relu))
            
        # construct and return sequential module
        ### list layers에 들어있는 module들을 수행하기 전에 input의 mean, std로 scale후, module을 수행하고, 다시 rescale.
        if input_scaling_mode == "scaling":
            reconstruct = InputScalingSequential(
                quantile, quantile_scale, OrderedDict(layers)
            )
        elif input_scaling_mode == "norming":
            reconstruct = InputNormingSequential((0, 2, 3), OrderedDict(layers))
        else:
            reconstruct = nn.Sequential(OrderedDict(layers))
        return reconstruct # sequential로 모델구성.