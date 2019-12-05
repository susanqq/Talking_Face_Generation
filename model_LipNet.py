import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import random

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


class LipEncoder(nn.Module):
    def __init__(self, embeding_size, img_channel):
        super(LipEncoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(img_channel, 16, kernel_size=5, stride=1, padding=2),
                                   nn.BatchNorm2d(16),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2) ) # (batch_size, 16, 20, 20)
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2) ) # (batch_size, 32, 10, 10)
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.MaxPool2d(kernel_size=2, stride=2) ) # (batch_size, 64, 5, 5)
        # reshape to (batch_size, 64x5x5)
        self.fc1 = nn.Sequential(nn.Linear(1600, embeding_size),nn.BatchNorm1d(embeding_size))

    def forward(self, inputs):  # input shape  (batch_size, 1, 40, 40)
        out = self.conv1(inputs)
        out = self.conv2(out)
        out = self.conv3(out)  # (batch_size, 64, 5, 5)
        # reshape to (batch_size, 64x5x5)
        out = out.contiguous().view(out.shape[0], out.shape[1]*out.shape[2]*out.shape[3])

        out = self.fc1(out)
        return out


class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type, num_layers):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = num_layers
        if rnn_type=='GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True)
        elif rnn_type=='LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, inputs, hidden):

        with torch.backends.cudnn.flags(enabled=False):
            output, hidden = self.rnn(inputs, hidden)

        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)

    def init_weights(self):
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

# image_inputs shape: batch_size, seq_len, c, h, w
class LipReadModel(nn.Module):
    def __init__(self, embeding_size, hidden_size, img_channel, num_layers, num_classes, encoder_type, rnn_type):
        super(LipReadModel, self).__init__()

        if encoder_type=='fc':
            self.encoder = LipEncoder(embeding_size, img_channel)
        if rnn_type=='GRU':
            self.rnn = RNNModel(embeding_size, hidden_size, 'GRU', num_layers)
        elif rnn_type=='LSTM':
            self.rnn = RNNModel(embeding_size, hidden_size, 'LSTM', num_layers)

        self.img_channel = img_channel
        self.encoder_type = encoder_type
        self.fc = nn.Linear(hidden_size, num_classes)


    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm1d):
                layer.eval()


    def forward(self, inputs, clip_range, lip_coords):   # (batch_size, seq_len, 3, H, W)
        # clip inputs to mouth region
        batch_size = inputs.shape[0]

        clip_len = int(clip_range[0][1] - clip_range[0][0])
        lip_size = int(lip_coords[0][0][2] - lip_coords[0][0][0])
        if lip_size!=40:
            print('********* here ***********')
        lip_inputs = inputs.new_zeros(batch_size, clip_len, self.img_channel, lip_size, lip_size)

        for i in range(batch_size):
            start_frame_id = clip_range[i][0]
            end_frame_id = clip_range[i][1]
            for j in range(clip_len):
                x1, y1, x2, y2 = lip_coords[i][j][0], lip_coords[i][j][1], lip_coords[i][j][2], lip_coords[i][j][3]
                if x2-x1 > lip_size:
                    x2 -= (x2-x1)-lip_size
                if x2-x1 < lip_size:
                    x2 += lip_size-(x2-x1)
                if y2-y1 > lip_size:
                    y2 -= (y2-y1)-lip_size
                if y2-y1 < lip_size:
                    y2 += lip_size-(y2-y1)

                
                if lip_inputs[i,j,:,:,:].shape != inputs[i, start_frame_id+j,:,y1:y2, x1:x2].shape:
                    print(lip_inputs.shape, inputs[i, start_frame_id+j,:,y1:y2, x1:x2].shape)
                    print(x1, y1, x2, y2)
                    
                lip_inputs[i,j,:,:,:] = inputs[i, start_frame_id+j,:,y1:y2, x1:x2]

         # reshape inputs to (seq_len*batch_size, ...)
        lip_inputs = lip_inputs.contiguous().view(batch_size*clip_len, lip_inputs.shape[2], lip_inputs.shape[3], lip_inputs.shape[4])

        embedding = self.encoder(lip_inputs) # (seq_len*batch_size, ...)

        # reshape to (batch_size, seq_len, ...)
        if self.encoder_type=='FCN':
            embedding = embedding.contiguous().view(batch_size, clip_len, embedding.shape[1],  embedding.shape[2],  embedding.shape[3])
        else:
            embedding = embedding.contiguous().view(batch_size, clip_len, embedding.shape[1])

        # fed to rnn
        hidden = self.rnn.init_hidden(batch_size)
        rnn_output, _ = self.rnn(embedding, hidden) # (batch_size, seq_len, hidden_size)
        output = self.fc(rnn_output[:,-1,:])
#        output = F.softmax(output, dim=1)

        return output




class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, inputs, gt):
        return self.criterion(inputs, gt)









