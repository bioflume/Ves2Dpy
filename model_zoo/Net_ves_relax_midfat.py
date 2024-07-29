
import torch
import torch.nn as nn

class Evo_Block(nn.Module):
    def __init__(self):
        super(Evo_Block, self).__init__()
        self.conv1_1 = nn.Conv1d(2, 10, kernel_size=7, stride=2, padding=3, padding_mode='circular')
        self.conv1_2 = nn.Conv1d(2, 10, kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular')
        self.bn1 = nn.BatchNorm1d(num_features=20)
        self.relu = nn.ReLU(inplace=True)
        self.conv2_1 = nn.Conv1d(20, 14, kernel_size=7, stride=2, padding=3, padding_mode='circular')
        self.conv2_2 = nn.Conv1d(20, 14, kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular')
        self.bn2 = nn.BatchNorm1d(num_features=28)
        self.deconv1 = nn.ConvTranspose1d(28, 24, kernel_size=10, stride=2, padding=4)
        self.bn3 = nn.BatchNorm1d(num_features=24)
        self.deconv2 = nn.ConvTranspose1d(24, 24, kernel_size=10, stride=2, padding=4)
        self.bn4 = nn.BatchNorm1d(num_features=24)
        self.mix = nn.Conv1d(24, 2, kernel_size=3, stride=1, padding=1)

    def forward(self, input):

        out_1_1 = self.relu(self.conv1_1(input))
        out_1_2 = self.relu(self.conv1_2(input))
        out = torch.concat((out_1_1, out_1_2), dim=1)
        out = self.bn1(out)

        out_2_1 = self.relu(self.conv2_1(out))
        out_2_2 = self.relu(self.conv2_2(out))
        out = torch.concat((out_2_1, out_2_2), dim=1)
        out = self.bn2(out)

        out = self.relu(self.deconv1(out))
        out = self.bn3(out)
        out = self.relu(self.deconv2(out))
        out = self.bn4(out)

        out = self.mix(out)

        out = out + input
        
        return out
    
class Net_ves_midfat(nn.Module):
    def __init__(self, num_blocks):
        super(Net_ves_midfat, self).__init__()
        self.layer = self.make_layer(num_blocks)

    def make_layer(self, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(Evo_Block())
        return nn.Sequential(*layers)
    
    def forward(self, input):
        return self.layer(input)

