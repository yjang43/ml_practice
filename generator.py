# ResNet Generator
import torch
import torch.nn as nn


# ResNet Block
class ResNetBlock(nn.Module):
    def __init__(self, fm):
        super().__init__()
        self.conv1 = nn.Conv2d(fm, fm, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(fm)
        
        self.conv2 = nn.Conv2d(fm, fm, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(fm)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + x
        out = self.relu(out)
        return out



# ResNet
class ResNet(nn.Module):
    def __init__(self, input_nc, output_nc, fm, n_blocks=6):
        super().__init__()
        net = [nn.ReflectionPad2d(3),
               nn.Conv2d(input_nc, fm, 7, 1, 0, bias=False),
               nn.BatchNorm2d(fm),
               nn.ReLU()]
        
        # downsample twice
        n_sampling = 2
        for i in range(n_sampling):
            net += [nn.Conv2d(fm * (2 ** i), fm * (2 ** (i + 1)), 3, 2, 1),
                    nn.BatchNorm2d(fm * (2 ** (i + 1))),
                    nn.ReLU()]
        
        # resnet block
        for i in range(n_blocks):
            net += [ResNetBlock(fm * (2 ** n_sampling))]
        
        # upsample twice
        for i in range(n_sampling):
            net += [nn.ConvTranspose2d(fm * (2 ** (n_sampling - i)), fm * (2 ** (n_sampling - i - 1)), 3, 2, 1, 1, bias=False),
                      nn.BatchNorm2d(fm * (2 ** (n_sampling - i - 1))),
                      nn.ReLU()]
        
        net += [nn.ReflectionPad2d(3),
                nn.Conv2d(fm, output_nc, 7, 1, 0, bias=False),
                nn.Sigmoid()]
        
        self.net = nn.Sequential(*net)
        
        
    def forward(self, x):
        return self.net(x)