import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, droprate=0.0):
        super(Block, self).__init__()
        self.stride = stride
        self.droprate = droprate
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 stride=stride, padding=0, bias=False)
    def forward(self, x):
        shortcut = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.droprate > 0:
            x = F.dropout(x, p=self.droprate)
        shortcut = self.conv1x1(shortcut)
    
        x += shortcut
        x = self.relu(x)
        
        return x
        
class WideResNet(nn.Module):
    def __init__(self, num_layers, widen_factor, block, num_classes=10, droprate=0.0):
        super(WideResNet, self).__init__()
        self.num_layers = num_layers
        self.droprate = droprate
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, 
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        
        self.layer1 = self.get_layers(block, 16, 16 * widen_factor, 1)
        self.layer2 = self.get_layers(block, 16 * widen_factor, 32 * widen_factor, 2)
        self.layer3 = self.get_layers(block, 32 * widen_factor, 64 * widen_factor, 2)
        
        self.avg_pool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(64 * widen_factor, num_classes)
        
    def get_layers(self, block, in_channels, out_channels, stride):
        layers = []
        
        for i in range(self.num_layers):
            if i == 0:
                layers.append(block(in_channels, out_channels, stride, self.droprate))
                continue
            layers.append(block(out_channels, out_channels, 1, self.droprate))
            
        return nn.Sequential(*layers)
                          
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

def wide_resnet(num_layers=5, widen_factor=2, num_classes=10, droprate=0.0):
    return WideResNet(num_layers, widen_factor, Block, num_classes, droprate)