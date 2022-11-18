import torch
import torch.nn as nn
import torch.nn.functional as F

'''
# ====================
# DNUNet
# ====================
'''
            
class Block(nn.Module):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> 1x1 Conv -> ReLU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); Linear -> ReLU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
    """
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=False) # depthwise conv
        self.pwconv1 = nn.Linear(dim, 4 * dim, bias=False) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.ReLU(True)
        self.pwconv2 = nn.Linear(4 * dim, dim, bias=False)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x


class DNUNet(nn.Module):
    def __init__(self, in_nc=1, out_nc=1, nc=[96, 192, 384, 768], nb=4):
        super(DNUNet, self).__init__()
        
        self.conv1 = nn.Conv2d(nc[0], nc[0], 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(nc[0], in_nc, 1, 1, 0, bias=False)
        self.conv3 = nn.Conv2d(in_nc, nc[0], 1, 1, 0, bias=False)
        self.conv4 = nn.Conv2d(in_nc, nc[0], 1, 1, 0, bias=False)
        self.conv5 = nn.Conv2d(2*nc[0], nc[0], 1, 1, 0, bias=False)
        
        self.m_head = nn.Conv2d(2*in_nc+1, nc[0], 3, 1, 1, bias=False)

        self.en_level1 = nn.Sequential(*[Block(nc[0]) for _ in range(nb)])
        self.m_down1 = nn.Conv2d(nc[0], nc[1], 2, 2, 0, bias=False)
        self.en_level2 = nn.Sequential(*[Block(nc[1]) for _ in range(nb)])
        self.m_down2 = nn.Conv2d(nc[1], nc[2], 2, 2, 0, bias=False)
        self.en_level3 = nn.Sequential(*[Block(nc[2]) for _ in range(nb)])
        self.m_down3 = nn.Conv2d(nc[2], nc[3], 2, 2, 0, bias=False)

        self.m_body  = nn.Sequential(*[Block(nc[3]) for _ in range(nb)])

        self.m_up3 = nn.ConvTranspose2d(nc[3], nc[2], 2, 2, 0, bias=False)
        self.de_level3 = nn.Sequential(*[Block(nc[2]) for _ in range(nb)])
        self.m_up2 = nn.ConvTranspose2d(nc[2], nc[1], 2, 2, 0, bias=False)
        self.de_level2 = nn.Sequential(*[Block(nc[1]) for _ in range(nb)])
        self.m_up1 = nn.ConvTranspose2d(nc[1], nc[0], 2, 2, 0, bias=False)
        self.de_level1 = nn.Sequential(*[Block(nc[0]) for _ in range(nb)])

        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x0, img_NL, Curv):
        X0 = torch.cat((x0, img_NL, Curv), dim=1)

        x_head = self.m_head(X0)
        x_en1 = self.en_level1(x_head)
        
        Fin = x_en1
        F1 = self.conv1(Fin)
        img = self.conv2(Fin) + x0
        F2 = torch.sigmoid(self.conv3(img))
        F3 = torch.sigmoid(self.conv4(Curv))
        XC = torch.cat((F2, F3), dim=1)
        F4 = self.conv5(XC)
        F1 = F1*F4
        Fout = F1+Fin
        
        x_down1 = self.m_down1(x_en1)
        x_en2 = self.en_level2(x_down1)
        x_down2 = self.m_down2(x_en2)
        x_en3 = self.en_level3(x_down2)
        x_down3 = self.m_down3(x_en3)
        x_body = self.m_body(x_down3)
        x_up3 = self.m_up3(x_body)
        x_de3 = self.de_level3(x_up3+x_en3)
        x_up2 = self.m_up2(x_de3)
        x_de2 = self.de_level2(x_up2+x_en2)
        x_up1 = self.m_up1(x_de2)
        x_de1 = self.de_level1(x_up1+Fout)
        x = self.m_tail(x_de1)+x0
        
        return x


if __name__ == '__main__':
    x = torch.rand(1,3,256,256)
    net = DNUNet()
    net.eval()
    with torch.no_grad():
        y = net(x)

