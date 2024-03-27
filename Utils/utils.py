import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import time as tm


def round(f):
    return math.ceil(f / 2.) * 2

def num_param(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

def trace_net(net, inp, save_pth="traced_model.pt"):
    traced_script_module = torch.jit.trace(
        net, inp, strict=True)
    traced_script_module.save(save_pth)

class MultiBatchNorm2d(nn.Module):
    def __init__(self, n1, n2, num_branch=None):
        super().__init__()
        self.num_branch = num_branch
        if num_branch is None:
            self.b1 = nn.BatchNorm2d(n1)
            self.b2 = nn.BatchNorm2d(n2)
        else:
            assert n2 is None
            self.b = nn.ModuleList(
                [nn.BatchNorm2d(n1) for _ in range(num_branch)])

    def forward(self, x):
        if self.num_branch is None:
            x1, x2 = x
            x1 = self.b1(x1)
            x2 = self.b2(x2)
            out = (x1, x2)
        else:
            out = []
            for _x, _b in zip(x, self.b):
                out.append(_b(_x))

        return out


class Concat2d(nn.Module):
    def __init__(self, shuffle=False):
        super().__init__()
        self.shuffle = shuffle

    def forward(self, x):
        if self.shuffle:
            b, _, h, w = x[0].shape
            x = [_x.unsqueeze(1) for _x in x]
            out = torch.cat(x, 1)
            out = out.transpose(1, 2)
            out = torch.reshape(out, (b, -1, h, w))
        else:
            out = torch.cat(x, 1)
        return out


class ReLU2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        x1, x2 = x
        x1 = self.relu(x1)
        x2 = self.relu(x2)
        return (x1, x2)


"""
source:
https://github.com/moskomule/senet.pytorch/blob/23839e07525f9f5d39982140fccc8b925fe4dee9/senet/se_module.py#L4-L19

"""
class SELayer(nn.Module):
    def __init__(self, channel, out_channel=None, reduction=16, version=1):
        super(SELayer, self).__init__()
        self.version = version
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        if out_channel is None:
            out_channel = channel
        self.channel = channel
        self.out_channel = out_channel
        if version == 1:
            self.fc = nn.Sequential(
                nn.Linear(channel, out_channel // reduction, bias=False),
                nn.ReLU(inplace=True),
                nn.Linear(out_channel // reduction, out_channel, bias=False),
                nn.Sigmoid()
            )
        elif version == 2:
            reduction = reduction // 2
            self.fc = nn.Sequential(
                nn.AvgPool1d(reduction),
                nn.Linear(channel // reduction, out_channel, bias=False),
                nn.Sigmoid()
            )
        else:
            assert False, version

    def forward(self, x, x2=None):
        if x2 is None:
            assert self.out_channel == self.channel
            x2 = x
        b, c, _, _ = x.size()
        b, c2, _, _ = x2.size()
        assert c == self.channel
        assert c2 == self.out_channel

        y = self.avg_pool(x).view(b, c)
        if self.version == 2:
            y = y.view(b, 1, c)
        y = self.fc(y).view(b, c2, 1, 1)
        return x2 * y.expand_as(x2)


class SE1(nn.Module):
    # Squeeze-and-excitation block in https://arxiv.org/abs/1709.01507
    # ch_in, ch_out, number, shortcut, groups, expansion
    def __init__(self, c_in, c_out, n=1, shortcut=True,  g=1, e=0.5, ver=1):
        super(SE1, self).__init__()
        self.ver = ver
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.cvsig = ConvSig(c_in, c_out, 1, 1, g=g)

    def forward(self, x):
        x = self.cvsig(self.avg_pool(x))
        if self.ver == 2:
            x = 2 * x
        return x

def update_bn(loader, model, total_imgs=1000, device=None):
    r"""Updates BatchNorm running_mean, running_var buffers in the model.
    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Arguments:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be trasferred to
            :attr:`device` before being passed into :attr:`model`.
    Example:
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)
    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    # print(model)

    momenta = {}
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.running_mean = torch.zeros_like(module.running_mean)
            module.running_var = torch.ones_like(module.running_var)
            momenta[module] = module.momentum

    if not momenta:
        return

    was_training = model.training
    model.train()
    for module in momenta.keys():
        module.momentum = None
        module.num_batches_tracked *= 0

    if device != None:
        model.to(device)

    num_images_total = 0

    for i, data in tqdm.tqdm(enumerate(loader), total = total_imgs):
        if i*loader.batch_size >= total_imgs:
            break
        img = data['img']
        img = img.to(device)
        model(img)

    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    model.train(was_training)

    print("update_bn is completed successfully, total_imgs = ", total_imgs)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class ConvSig(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSig, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.Sigmoid() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class ConvSqu(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super(ConvSqu, self).__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.conv(x))

    def fuseforward(self, x):
        return self.act(self.conv(x))

# source: https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py
class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size,
                              padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(
            y.squeeze(-1).transpose(-1, -2)
            ).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return y

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1):
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

def fuse_model(m):
    prev_previous_type = nn.Identity()
    prev_previous_name = ''
    previous_type = nn.Identity()
    previous_name = ''
    for name, module in m.named_modules():
        if prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d and type(module) == nn.ReLU:
            print("FUSED ", prev_previous_name, previous_name, name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name, name], inplace=True)
        elif prev_previous_type == nn.Conv2d and previous_type == nn.BatchNorm2d:
            print("FUSED ", prev_previous_name, previous_name)
            torch.quantization.fuse_modules(m, [prev_previous_name, previous_name], inplace=True)
        elif previous_type == nn.Conv2d and type(module) == nn.ReLU:
            print("FUSED ", previous_name, name)
            #torch.quantization.fuse_modules(m, [previous_name, name], inplace=True)

        prev_previous_type = previous_type
        prev_previous_name = previous_name
        previous_type = type(module)
        previous_name = name

def sparsity(model):
    # Return global model sparsity
    a, b = 0., 0.
    for p in model.parameters():
        a += p.numel()
        b += (p == 0).sum()
    return b / a


def prune(model, amount=0.3):
    # Prune model to requested global sparsity
    import torch.nn.utils.prune as prune
    print('Pruning model... ', end='')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            prune.l1_unstructured(m, name='weight', amount=amount)  # prune
            prune.remove(m, 'weight')  # make permanent
    print(' %.3g global sparsity' % sparsity(model))


def fuse_conv_and_bn(conv, bn):
    # https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    with torch.no_grad():
        # init
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)

        # prepare filters
        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        # prepare spatial bias
        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv


def model_info(model, input=torch.zeros(1, 3, 224, 224), verbose=False):
    # Plots a line-by-line description of a PyTorch model
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        print('%5s %40s %9s %12s %20s %10s %10s' % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %
                  (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))

    try:  # FLOPS
        from thop import profile
        print("try thop")
        flops = profile((model), inputs=(input,), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GFLOPS' % (flops)  # 224x224 FLOPS
    except:
        fs = ''

    print('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # reshape
    x = x.view(batchsize, groups,
               channels_per_group, height, width)

    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(batchsize, -1, height, width)

    return x