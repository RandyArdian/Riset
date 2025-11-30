import torch
import torch.nn as nn

class Conv(nn.Module):
    def __init__(self, c1,c2):
        super().__init__()
        self.conv=nn.Conv2d(c1, c2, kernel_size=3, padding=1, stride=1, bias=True)
        self.norm=nn.BatchNorm2d(c2)
        self.aktifasi=nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.aktifasi(self.norm(self.conv(x)))
    
class TCMUNiXt(nn.Module):
    def __init__(self, c1, c2, k=3):
        super().__init__()
        self.dw= nn.Conv2d(c1, c1, kernel_size=(k,k), groups=c1, padding=(k//2,k//2))
        self.gelu=nn.GELU()
        self.batchnorm= nn.BatchNorm2d(c1)
        self.batchnorm2=nn.BatchNorm2d(4*c1)
        self.pointwise=nn.Conv2d(c1, 4 * c1, kernel_size=1)
        self.pointwise2=nn.Conv2d(4*c1, c1, kernel_size=1)
        self.up=Conv(c1*4, c2)
        
    def forward(self, x):
        #membuat residual 
        X=x
        # Tahap 1
        x=self.batchnorm(self.gelu(self.dw(x)))
        x=X+x
        #Tahap 2
        x=self.batchnorm2(self.gelu(self.pointwise(x)))
        #Tahap 3
        x=self.batchnorm(self.gelu(self.pointwise2(x)))
        #tahap 4
        x=self.batchnorm2(self.gelu(self.pointwise(x)))
        x=self.up(x)
        return x