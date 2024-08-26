import torch
import torch.nn as nn
import math
# from Net_ves_adv_fft import Net_ves_adv_fft

class Starter_Block(nn.Module):
    def __init__(self, factor, ch, rep):
        super(Starter_Block, self).__init__()
        self.rep = rep
        self.ch = ch
        conv1_out_ch = math.floor(9*factor)
        self.conv1_1 = nn.Conv1d(2*rep, conv1_out_ch*rep, kernel_size=7, stride=2, padding=3, padding_mode='circular', groups=rep)
        self.conv1_2 = nn.Conv1d(2*rep, conv1_out_ch*rep, kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular', groups=rep)
        self.bn1 = nn.BatchNorm1d(num_features=2*conv1_out_ch*rep)
        self.relu = nn.ReLU(inplace=True)
        conv2_out_ch = math.floor(10*factor)
        self.conv2_1 = nn.Conv1d(2*conv1_out_ch*rep, conv2_out_ch*rep, kernel_size=7, stride=2, padding=3, padding_mode='circular', groups=rep)
        self.conv2_2 = nn.Conv1d(2*conv1_out_ch*rep, conv2_out_ch*rep, kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular', groups=rep)
        deconv_ch = 2*conv2_out_ch*rep
        self.bn2 = nn.BatchNorm1d(num_features=deconv_ch)
        self.deconv1 = nn.ConvTranspose1d(deconv_ch, deconv_ch, kernel_size=10, stride=2, padding=4, groups=rep)
        self.bn3 = nn.BatchNorm1d(num_features=deconv_ch)
        self.deconv2 = nn.ConvTranspose1d(deconv_ch, deconv_ch, kernel_size=10, stride=2, padding=4, groups=rep)
        self.bn4 = nn.BatchNorm1d(num_features=deconv_ch)

        self.mix = nn.Conv1d(deconv_ch, ch*rep, kernel_size=3, stride=1, padding=1, groups=rep)

    def forward(self, input):
        bs = input.shape[0]
        out_1_1 = self.relu(self.conv1_1(input))
        out_1_2 = self.relu(self.conv1_2(input))
        ch = out_1_1.shape[1]//self.rep
        # features = [torch.cat((out_1_1[:, (ch)*i:(ch)*(i+1)],
        #                        out_1_2[:, (ch)*i:(ch)*(i+1)],),dim=1)  for i in range(self.rep)]
        # out = torch.cat(tuple(features), dim=1)
        out = torch.cat((out_1_1.reshape(bs, self.rep, ch, -1), out_1_2.reshape(bs, self.rep, ch, -1)), dim=2)
        out = out.reshape(bs, 2*ch*self.rep, -1)
        out = self.bn1(out)

        out_2_1 = self.relu(self.conv2_1(out))
        out_2_2 = self.relu(self.conv2_2(out))
        ch = out_2_1.shape[1]//self.rep
        # features = [torch.cat((out_2_1[:, (ch)*i:(ch)*(i+1)],
        #                        out_2_2[:, (ch)*i:(ch)*(i+1)],),dim=1)  for i in range(self.rep)]
        # out = torch.cat(tuple(features), dim=1)
        out = torch.cat((out_2_1.reshape(bs, self.rep, ch, -1), out_2_2.reshape(bs, self.rep, ch, -1)), dim=2)
        out = out.reshape(bs, 2*ch*self.rep, -1)
        out = self.bn2(out)
        

        out = self.relu(self.deconv1(out))
        out = self.bn3(out)
        out = self.relu(self.deconv2(out))
        out = self.bn4(out)

        out = self.mix(out)
        # members = [torch.cat((out[:, self.ch*i:self.ch*(i+1)], input[:,[2*i+1]]),dim=1) for i in range(self.rep)]
        # out = torch.cat(tuple(members), dim=1)
        out = torch.cat((out.reshape(bs, self.rep, self.ch, -1), 
                         input[:, [2*i+1 for i in range(self.rep)]].reshape(bs, self.rep, 1, -1)), dim=2)
        out = out.reshape(bs, (self.ch+1)*self.rep, -1)
        
        return out

class Evo_Block(nn.Module):
    def __init__(self, factor, ch, rep):
        super(Evo_Block, self).__init__()
        self.rep = rep
        self.ch = ch
        conv1_out_ch = math.floor(8*factor)
        self.conv1_1 = nn.Conv1d(rep*(ch+1), conv1_out_ch*rep, kernel_size=7, stride=2, padding=3, padding_mode='circular', groups=rep)
        self.conv1_2 = nn.Conv1d(rep*(ch+1), conv1_out_ch*rep, kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular', groups=rep)
        self.bn1 = nn.BatchNorm1d(num_features=2*conv1_out_ch*rep)
        self.relu = nn.ReLU(inplace=True)
        conv2_out_ch = math.floor(10*factor)
        self.conv2_1 = nn.Conv1d(2*conv1_out_ch*rep, conv2_out_ch*rep, 
                                 kernel_size=7, stride=2, padding=3, padding_mode='circular', groups=rep)
        self.conv2_2 = nn.Conv1d(2*conv1_out_ch*rep, conv2_out_ch*rep, 
                                 kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular', groups=rep)
        deconv_ch = 2*conv2_out_ch*rep
        self.bn2 = nn.BatchNorm1d(num_features=deconv_ch)
        self.deconv1 = nn.ConvTranspose1d(deconv_ch, deconv_ch, kernel_size=10, stride=2, padding=4, groups=rep)
        self.bn3 = nn.BatchNorm1d(num_features=deconv_ch)
        self.deconv2 = nn.ConvTranspose1d(deconv_ch, deconv_ch, kernel_size=10, stride=2, padding=4, groups=rep)
        self.bn4 = nn.BatchNorm1d(num_features=deconv_ch)
        self.mix = nn.Conv1d(deconv_ch, ch*rep, kernel_size=3, stride=1, padding=1, groups=rep)

    def forward(self, input):
        bs = input.shape[0]
        
        # features = [input[:, (self.ch+1)*i:(self.ch+1)*(i+1)-1] for i in range(self.rep)]
        # residual = torch.cat(tuple(features), dim=1)
        ids = torch.concat([torch.arange((self.ch+1)*i, (self.ch+1)*(i+1)-1) for i in range(self.rep)])
        residual = input[:,ids]

        out_1_1 = self.relu(self.conv1_1(input))
        out_1_2 = self.relu(self.conv1_2(input))
        ch = out_1_1.shape[1]//self.rep
        # features = [torch.cat((out_1_1[:, (ch)*i:(ch)*(i+1)],
        #                        out_1_2[:, (ch)*i:(ch)*(i+1)],),dim=1)  for i in range(self.rep)]
        # out = torch.cat(tuple(features), dim=1)
        out = torch.cat((out_1_1.reshape(bs, self.rep, ch, -1), out_1_2.reshape(bs, self.rep, ch, -1)), dim=2)
        out = out.reshape(bs, 2*ch*self.rep, -1)
        out = self.bn1(out)

        out_2_1 = self.relu(self.conv2_1(out))
        out_2_2 = self.relu(self.conv2_2(out))
        ch = out_2_1.shape[1]//self.rep
        # features = [torch.cat((out_2_1[:, (ch)*i:(ch)*(i+1)],
        #                        out_2_2[:, (ch)*i:(ch)*(i+1)],),dim=1)  for i in range(self.rep)]
        # out = torch.cat(tuple(features), dim=1)
        out = torch.cat((out_2_1.reshape(bs, self.rep, ch, -1), out_2_2.reshape(bs, self.rep, ch, -1)), dim=2)
        out = out.reshape(bs, 2*ch*self.rep, -1)
        out = self.bn2(out)

        out = self.relu(self.deconv1(out))
        out = self.bn3(out)
        out = self.relu(self.deconv2(out))
        out = self.bn4(out)

        out = self.mix(out)

        out = out + residual

        # members = [torch.cat((out[:, self.ch*i:self.ch*(i+1)], input[:,[(self.ch+1)*(i+1)-1]]),dim=1) for i in range(self.rep)]
        # out_u = torch.cat(tuple(members), dim=1)
        out = torch.cat((out.reshape(bs, self.rep, self.ch, -1), 
                         input[:, [(self.ch+1)*(i+1)-1 for i in range(self.rep)]].reshape(bs, self.rep, 1, -1)), dim=2)
        out_u = out.reshape(bs, (self.ch+1)*self.rep, -1)
       
        return out_u

class End_Block(nn.Module):
    def __init__(self, factor, ch, rep):
        super(End_Block, self).__init__()
        self.ch = ch
        self.rep = rep
        conv1_out_ch = math.floor(8*factor)
        self.conv1_1 = nn.Conv1d(rep*(ch+1), conv1_out_ch*rep, kernel_size=7, stride=2, padding=3, padding_mode='circular', groups=rep)
        self.conv1_2 = nn.Conv1d(rep*(ch+1), conv1_out_ch*rep, kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular', groups=rep)
        self.bn1 = nn.BatchNorm1d(num_features=2*conv1_out_ch*rep)
        self.relu = nn.ReLU(inplace=True)
        conv2_out_ch = math.floor(10*factor)
        self.conv2_1 = nn.Conv1d(2*conv1_out_ch*rep, conv2_out_ch*rep, 
                                 kernel_size=7, stride=2, padding=3, padding_mode='circular', groups=rep)
        self.conv2_2 = nn.Conv1d(2*conv1_out_ch*rep, conv2_out_ch*rep, 
                                 kernel_size=7, stride=2, padding=6, dilation=2, padding_mode='circular', groups=rep)
        deconv_ch = 2*conv2_out_ch*rep
        self.bn2 = nn.BatchNorm1d(num_features=deconv_ch)
        self.deconv1 = nn.ConvTranspose1d(deconv_ch, deconv_ch, kernel_size=10, stride=2, padding=4, groups=rep)
        self.bn3 = nn.BatchNorm1d(num_features=deconv_ch)
        self.deconv2 = nn.ConvTranspose1d(deconv_ch, deconv_ch, kernel_size=10, stride=2, padding=4, groups=rep)
        self.bn4 = nn.BatchNorm1d(num_features=deconv_ch)
        self.mix = nn.Conv1d(deconv_ch, 2*rep, kernel_size=3, stride=1, padding=1, groups=rep)

    def forward(self, input):
        # residual = input[:,:-1]
        # mode = input[:,-1].reshape(-1,1,256)
        bs = input.shape[0]

        out_1_1 = self.relu(self.conv1_1(input))
        out_1_2 = self.relu(self.conv1_2(input))
        ch = out_1_1.shape[1]//self.rep
        # features = [torch.cat((out_1_1[:, (ch)*i:(ch)*(i+1)],
        #                        out_1_2[:, (ch)*i:(ch)*(i+1)],),dim=1)  for i in range(self.rep)]
        # out = torch.cat(tuple(features), dim=1)
        out = torch.cat((out_1_1.reshape(bs, self.rep, ch, -1), out_1_2.reshape(bs, self.rep, ch, -1)), dim=2)
        out = out.reshape(bs, 2*ch*self.rep, -1)
        out = self.bn1(out)

        out_2_1 = self.relu(self.conv2_1(out))
        out_2_2 = self.relu(self.conv2_2(out))
        ch = out_2_1.shape[1]//self.rep
        # features = [torch.cat((out_2_1[:, (ch)*i:(ch)*(i+1)],
        #                        out_2_2[:, (ch)*i:(ch)*(i+1)],),dim=1)  for i in range(self.rep)]
        # out = torch.cat(tuple(features), dim=1)
        out = torch.cat((out_2_1.reshape(bs, self.rep, ch, -1), out_2_2.reshape(bs, self.rep, ch, -1)), dim=2)
        out = out.reshape(bs, 2*ch*self.rep, -1)
        out = self.bn2(out)

        out = self.relu(self.deconv1(out))
        out = self.bn3(out)
        out = self.relu(self.deconv2(out))
        out = self.bn4(out)

        out = self.mix(out)
        return out


class Net_merge_advection(nn.Module):
    def __init__(self, num_blocks, factor, ch, rep):
        super(Net_merge_advection, self).__init__()
        self.factor = factor
        self.ch = ch
        self.rep = rep
        self.layer = self.make_layer(num_blocks-1, ch, rep)
        self.starter = Starter_Block(factor, ch, rep)
        self.end = End_Block(factor, ch, rep)

    def make_layer(self, num_blocks, ch, rep):
        layers = []
        for _ in range(num_blocks):
            layers.append(Evo_Block(self.factor, ch, rep))
        return nn.Sequential(*layers)
    
    def forward(self, input):
        out = self.layer(self.starter(input))
        ans = self.end(out)
        return ans

# model = Net_merge_advection(4,1.7,20,2)
# print(model(torch.randn(1,4,256)).shape)
        
        