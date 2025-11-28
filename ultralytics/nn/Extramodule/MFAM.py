import torch
import math
import torch.nn as nn


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    #default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act =  nn.SiLU() if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))


class DWConv(Conv):
    """Depth-wise convolution."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        """Initialize Depth-wise convolution with given parameters."""
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)  #gcd mencari FPB dari dua bilangan

class MFAM(nn.Module):
    def __init__(self, c1, c2, expand_ratio=1):
        super(MFAM, self).__init__()
        hidden_dim= c1 * expand_ratio
        # DWconv pertama
        self.cv1 = DWConv(c1, hidden_dim, k=3)
        # Dwconv kedua
        self.cv2 = DWConv(c1, hidden_dim, k=5)
        # path 3
        self.cv3= DWConv(c1, hidden_dim,k=(1,7))
        self.cv4= DWConv(hidden_dim, hidden_dim,k=(7,1))

        # path 4
        self.cv5= DWConv(c1, hidden_dim,k=(1,9))
        self.cv6= DWConv(hidden_dim, hidden_dim,k=(9,1))
        
        # Konvolusi terakhir
        self.cv7= Conv(hidden_dim, c2, k=1)


    def forward(self, x):
        identity=x
        path1= self.cv1(x)
        path2= self.cv2(x)
        path3= self.cv4(self.cv3(x))
        path4= self.cv6(self.cv5(x))
        add=path1+path2+path3+path4+identity
        return self.cv7(add)

