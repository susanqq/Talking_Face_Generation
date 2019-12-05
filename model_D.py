import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import model_LipNet as LipNet

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and hasattr(m, 'weight'):
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)
    # elif classname.find('GRU') != -1 or classname.find('LSTM') != -1:
    #     m.weight.data.normal_(0.0, 0.02)
    #     m.bias.data.fill_(0.01)
    else:
        print(classname)


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        return out


class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim), nn.ReLU(inplace=True)]
        conv_block += [nn.ReflectionPad2d(1), nn.Conv2d(dim, dim, kernel_size=3), nn.InstanceNorm2d(dim)]
        self.conv_blocks = nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_blocks(x)
        return out




class DiscriminatorFrame(nn.Module):
    def __init__(self):
        super(DiscriminatorFrame, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.InstanceNorm2d(16), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.InstanceNorm2d(32), nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True))
        self.conv5 = nn.Conv2d(128, 1, 1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class DiscriminatorVideoConv3D(nn.Module):
    def __init__(self, ndf=64):
        super(DiscriminatorVideoConv3D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(3, ndf, 3, stride=(1, 2, 2), padding=(0, 1, 1)), nn.BatchNorm3d(ndf), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv3d(ndf, ndf * 2, 3, stride=(1, 2, 2), padding=(0, 1, 1)), nn.BatchNorm3d(ndf * 2), nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(nn.Conv3d(ndf * 2, ndf * 4, 3, stride=(1, 2, 2), padding=(0, 1, 1)),nn.BatchNorm3d(ndf * 4), nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Conv3d(ndf * 4, 1, 3, stride=(1, 2, 2), padding=(0, 1, 1))

    def forward(self, inputs):
        out = self.conv1(inputs)
        # print(out.size())
        out = self.conv2(out)
        # print(out.size())
        out = self.conv3(out)
        # print(out.size())
        out = self.conv4(out)
        # print(out.size())
        return out

class DiscriminatorVideoConv2D(nn.Module):
    def __init__(self, num_of_frame, ndf=64):
        super(DiscriminatorVideoConv2D, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3*num_of_frame, ndf, 3, stride=2, padding=1), nn.InstanceNorm2d(64), nn.LeakyReLU(0.2, True))
        self.conv2 = nn.Sequential(nn.Conv2d(ndf, ndf*2, 3, stride=2, padding=1), nn.InstanceNorm2d(128), nn.LeakyReLU(0.2, True))
        self.conv3 = nn.Sequential(nn.Conv2d(ndf*2, ndf*4, 3, stride=2, padding=1), nn.InstanceNorm2d(256), nn.LeakyReLU(0.2, True))
        self.conv4 = nn.Sequential(nn.Conv2d(ndf*4, ndf*8, 3, stride=2, padding=1), nn.InstanceNorm2d(512), nn.LeakyReLU(0.2, True))
        self.conv5 = nn.Conv2d(ndf*8, 1, 1)

    def forward(self, inputs):
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        return out


class DiscriminatorVideo(nn.Module):
    def __init__(self, config):
        super(DiscriminatorVideo, self).__init__()
        if config.discriminator_v == 'video_3D':
            self.discriminator = DiscriminatorVideoConv3D(ndf=64)
        elif config.discriminator_v =='video_2D':
            self.discriminator = DiscriminatorVideoConv2D(config.num_frames_D, ndf=64)

    def forward(self, inputs, num_frames_D, clip_range=None): # (batch_size, seq_len, c, h, w)

        batch_size = inputs.shape[0]
        clip_inputs = inputs.new_zeros(batch_size, num_frames_D, inputs.shape[2], inputs.shape[3], inputs.shape[4])

        for i in range(batch_size):
            start_idx = clip_range[i][0]
            end_dix = clip_range[i][1]
            if (end_dix-start_idx)==num_frames_D:
                clip_inputs[i,:,:,:,:] = inputs[i, start_idx:end_dix,:,:,:]
            else:
                print("sequence length less than num_frames_D")
                diff = num_frames_D-(end_dix-start_idx)
                padding = inputs[i,-1,:,:,:].unsqueeze(0).repeat(1, diff, 1, 1, 1)
                clip_inputs[i,:(end_dix-start_idx),:,:,:] = inputs[i, start_idx:end_dix,:,:,:]
                clip_inputs[i,(end_dix-start_idx):,:,:,:] = padding

        clip_inputs = clip_inputs.permute(0, 2, 1, 3, 4)
        return self.discriminator(clip_inputs)


class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.discriminator = DiscriminatorFrame()

    def forward(self, inputs):
        if len(inputs.shape)==5:
            inputs = inputs.contiguous().view(inputs.shape[0]*inputs.shape[1], inputs.shape[2], inputs.shape[3], inputs.shape[4])
        return self.discriminator(inputs)



class DiscriminatorLip(nn.Module):
    def __init__(self, config):
        super(DiscriminatorLip, self).__init__()
        self.discriminator = LipNet.LipReadModel(512, 512, img_channel=3, num_layers=2, num_classes=500, encoder_type='fc', rnn_type='LSTM')

    def forward(self, inputs, clip_range=None, lip_coords=None):
        return self.discriminator(inputs, clip_range, lip_coords)



