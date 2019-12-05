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


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
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


class AudioEncoder(nn.Module):
    def __init__(self, num_output_length, if_tanh=False):
        super(AudioEncoder, self).__init__()
        self.if_tanh = if_tanh
        # the input map is 1 x 12 x 35
        self.block1 = BasicBlock(1, 16, kernel_size=3, stride=1) # 16 x 12 x 35
        self.block2 = BasicBlock(16, 32, kernel_size=3, stride=2) # 32 x 6 x 18
        self.block3 = BasicBlock(32, 64, kernel_size=3, stride=1) # 64 x 6 x 18
        self.block4 = BasicBlock(64, 128, kernel_size=3, stride=1) # 128 x 6 x 18
        self.block5 = BasicBlock(128, 256, kernel_size=3, stride=2) # 256 x 3 x 9
        # self.fc1 = nn.Linear(6912, 512)
        # self.batch_norm = nn.BatchNorm2d(512)
        self.fc1 = nn.Sequential(nn.Linear(6912, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(512, num_output_length)

    def forward(self, inputs):
        out = self.block1(inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.contiguous().view(out.shape[0], -1)
        # out = F.relu(self.batch_norm(self.fc1(out)))
        out = self.fc1(out)
        out = self.fc2(out)
        if self.if_tanh:
          out = F.tanh(out)
        return out

class AudioEncoderBMVC(nn.Module):
    def __init__(self, num_output_length, if_tanh=False):
        super(AudioEncoderBMVC, self).__init__()
        self.if_tanh = if_tanh
        self.block1 = BasicBlock(1, 32, kernel_size=3, stride=1) # 32 x 12 x 35
        self.block2 = BasicBlock(32, 64, kernel_size=3, stride=[1,2]) # 64 x 12 x 18
        self.block3 = BasicBlock(64, 128, kernel_size=3, stride=1) # 128 x 12 x 18
        self.block4 = BasicBlock(128, 128, kernel_size=3, stride=1) # 128 x 12 x 18
        self.block5 = BasicBlock(128, 256, kernel_size=3, stride=2) # 256 x 6 x 9
        self.fc1 = nn.Sequential(nn.Linear(13824, 1024), nn.ReLU(inplace=True))
        self.fc2 = nn.Linear(1024, num_output_length)
    def forward(self, inputs):
        out = self.block1(inputs)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out = self.block5(out)
        out = out.contiguous().view(out.shape[0], -1)
        # out = F.relu(self.batch_norm(self.fc1(out)))
        out = self.fc1(out)
        out = self.fc2(out)
        if self.if_tanh:
          out = F.tanh(out)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, size_image, num_output_length, if_tanh=False):
        super(ImageEncoder, self).__init__()
        self.if_tanh = if_tanh
        self.conv1 = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 128, 5, stride=2, padding=2), nn.ReLU(inplace=True))
        size_mini_map = self.get_size(size_image, 4)
        self.fc = nn.Linear(size_mini_map*size_mini_map*128, num_output_length)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, inputs):
        img_e_conv1 = self.conv1(inputs)
        img_e_conv2 = self.conv2(img_e_conv1)
        img_e_conv3 = self.conv3(img_e_conv2)
        img_e_conv4 = self.conv4(img_e_conv3)
        img_e_fc_5 = img_e_conv4.contiguous().view(img_e_conv4.shape[0], -1)
        img_e_fc_5 = self.fc(img_e_fc_5)
        if self.if_tanh:
            img_e_fc_5 = F.tanh(img_e_fc_5)
        return img_e_fc_5, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4


# encoder is fully convolutional
class ImageEncoderFCN(nn.Module):
    def __init__(self, size_image, num_output_length, if_tanh=False):
        super(ImageEncoderFCN, self).__init__()
        self.if_tanh = if_tanh
        self.conv1 = BasicBlock(3, 64, kernel_size=3, stride=2)# nn.Sequential(nn.Conv2d(3, 64, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv2 = BasicBlock(64, 128, kernel_size=3, stride=2)# nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv3 = BasicBlock(128, 256, kernel_size=3, stride=2)# nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True))
        self.conv4 = BasicBlock(256, 512, kernel_size=3, stride=2)# nn.Sequential(nn.Conv2d(256, 512, 3, stride=2, padding=1), nn.ReLU(inplace=True))

    def forward(self, inputs):
        img_e_conv1 = self.conv1(inputs) # /2
        img_e_conv2 = self.conv2(img_e_conv1) # /4
        img_e_conv3 = self.conv3(img_e_conv2) # /8
        img_e_conv4 = self.conv4(img_e_conv3) # /16
        if self.if_tanh:
            img_e_conv4 = F.tanh(img_e_conv4)
        return img_e_conv4, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4



class ImageDecoder(nn.Module):
    def __init__(self, size_image, input_dim):
        super(ImageDecoder, self).__init__()
        self.size_mini_map = self.get_size(size_image, 4)
        self.fc = nn.Linear(input_dim, self.size_mini_map*self.size_mini_map*256)
        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(384, 196, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(260, 128, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(160, 80, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv4 = nn.Sequential(nn.ConvTranspose2d(96, 48, 5, stride=2, padding=2, output_padding=1), nn.ReLU(inplace=True))
        self.dconv5 = nn.Sequential(nn.Conv2d(48, 16, 5, stride=1, padding=2), nn.ReLU(inplace=True))
        self.dconv6 = nn.Conv2d(16, 3, 5, stride=1, padding=2)

    def get_size(self, size_image, num_layers):
        return int(size_image/2**num_layers)

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        # out = torch.cat([img_z, audio_z], dim=1) # (batch_size, input_dim)
        out = self.fc(concat_z)
        # reshape 256 x 7 x 7
        out = out.contiguous().view(out.shape[0],  256, self.size_mini_map, self.size_mini_map)
        out = F.relu(out, inplace=True)
        # concate (256+128) x 7x7
        out = torch.cat([out, img_e_conv4], dim=1)
        out = self.dconv1(out)
        # concate (196+64) x 14x14
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.dconv2(out)
        # concate (128+32) x 28x28
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.dconv3(out)
        # concate (80+16) x 56x56
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.dconv4(out)
        out = self.dconv5(out)
        out = self.dconv6(out)
        return F.tanh(out)


class ImageDecoderResidual(nn.Module):
    def __init__(self, size_image, input_dim):
        super(ImageDecoderResidual, self).__init__()
        self.fuse = nn.Sequential(nn.Conv2d(input_dim, 512, 3, stride=1, padding=1), nn.ReLU(True))
        self.resblocks = nn.Sequential(ResnetBlock(512))#, ResnetBlock(512), ResnetBlock(512))
        self.dconv1 = nn.Sequential(nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.conv1 = nn.Conv2d(512, 256, 3, stride=1, padding=1)
        self.dconv2 = nn.Sequential(nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.conv2 = nn.Conv2d(256, 128, 3, stride=1, padding=1)
        self.dconv3 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.conv3 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.dconv4 = nn.Sequential(nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.conv4 = nn.Conv2d(32, 3, 3, stride=1, padding=1)

    def forward(self, concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4):
        # audio_z = audio_z.unsqueeze(-1).unsqueeze(-1)
        # audio_z = audio_z.repeat(1,1, img_z.shape[2], img_z.shape[3])
        # z = torch.cat([img_z, audio_z], dim=1)
        z = self.fuse(concat_z)
        z = self.resblocks(z)
        out = self.dconv1(z)
        out = torch.cat([out, img_e_conv3], dim=1)
        out = self.conv1(out)
        out = self.dconv2(out)
        out = torch.cat([out, img_e_conv2], dim=1)
        out = self.conv2(out)
        out = self.dconv3(out)
        out = torch.cat([out, img_e_conv1], dim=1)
        out = self.conv3(out)
        out = self.dconv4(out)
        out = self.conv4(out)
        return F.tanh(out)


class LipGeneratorCNN(nn.Module):
    def __init__(self, audio_encoder_type, img_encoder_type, img_decoder_type, size_image, num_output_length, if_tanh):
        super(LipGeneratorCNN, self).__init__()
        if audio_encoder_type=='reduce':
            self.audio_encoder = AudioEncoder(num_output_length, if_tanh)
        elif audio_encoder_type=='bmvc':
            self.audio_encoder = AudioEncoderBMVC(num_output_length, if_tanh)
        if img_encoder_type=='reduce':
            self.image_encoder = ImageEncoder(size_image, num_output_length, if_tanh)
        elif img_encoder_type=='FCN':
            self.image_encoder = ImageEncoderFCN(size_image, num_output_length, if_tanh)
        if img_decoder_type=='reduce':
            self.image_decoder = ImageDecoder(size_image, 2*num_output_length)
        elif img_decoder_type=='residual':
            self.image_decoder = ImageDecoderResidual(size_image, 2*num_output_length)

        self.audio_encoder_type = audio_encoder_type
        self.img_encoder_type = img_encoder_type
        self.img_decoder_type = img_decoder_type

        # initialize weights
        self.audio_encoder.apply(weights_init)
        self.image_encoder.apply(weights_init)
        self.image_decoder.apply(weights_init)

    def forward(self, image_inputs, audio_inputs):
        audio_z = self.audio_encoder(audio_inputs)
        image_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4 = self.image_encoder(image_inputs)

        if self.img_encoder_type=='FCN':
            audio_z = audio_z.unsqueeze(-1).unsqueeze(-1)
            audio_z = audio_z.repeat(1,1, image_z.shape[2], image_z.shape[3])
        concat_z = torch.cat([image_z, audio_z], dim=1)

        G = self.image_decoder(concat_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)
        return G

    def model_type(self):
        return 'CNN'




class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type, num_layers=1):
        super(RNNModel, self).__init__()
        self.rnn_type = rnn_type
        self.nhid = hidden_size
        self.nlayers = num_layers
        if rnn_type=='GRU':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=1)

    def forward(self, inputs, hidden):

        output, hidden = self.rnn(inputs, hidden)
        return output, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, batch_size, self.nhid),
                    weight.new_zeros(self.nlayers, batch_size, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, batch_size, self.nhid)


class LipGeneratorRNN(nn.Module):
    def __init__(self, audio_encoder_type, img_encoder_type, img_decoder_type, rnn_type, size_image, num_output_length, hidden_size=1024, if_tanh=False):
        super(LipGeneratorRNN, self).__init__()
        if audio_encoder_type=='reduce':
            self.audio_encoder = AudioEncoder(num_output_length, if_tanh)
        elif audio_encoder_type=='bmvc':
            self.audio_encoder = AudioEncoderBMVC(num_output_length, if_tanh)
        if img_encoder_type=='reduce':
            self.image_encoder = ImageEncoder(size_image, num_output_length, if_tanh)
        elif img_encoder_type=='FCN':
            self.image_encoder = ImageEncoderFCN(size_image, num_output_length, if_tanh)
        if img_decoder_type=='reduce':
            self.image_decoder = ImageDecoder(size_image, hidden_size)
        elif img_decoder_type=='residual':
            self.image_decoder = ImageDecoderResidual(size_image, hidden_size)
        if rnn_type=='GRU':
            self.rnn = RNNModel(2*num_output_length, hidden_size, rnn_type)


        self.audio_encoder_type = audio_encoder_type
        self.img_encoder_type = img_encoder_type
        self.img_decoder_type = img_decoder_type

        # initialize weights
        self.audio_encoder.apply(weights_init)
        self.image_encoder.apply(weights_init)
        self.image_decoder.apply(weights_init)
        self.rnn.apply(weights_init)

    # image_inputs shape: batch_size, seq_len, c, h, w
    def forward(self, image_inputs, audio_inputs, valid_len, teacher_forcing_ratio=0.5):
        # reshape inputs to (seq_len*batch_size, ...)
        seq_len = image_inputs.shape[1]
        batch_size = image_inputs.shape[0]

        # G = image_inputs.new_zeros(batch_size, seq_len, image_inputs.shape[2], image_inputs.shape[3], image_inputs.shape[4])

        valid_l = valid_len.min()
        hidden = self.rnn.init_hidden(batch_size)
        for i in range(valid_l):
            image_batch = image_inputs[:,i,:,:,:] # (batch_size, c, h, w)
            audio_batch = audio_inputs[:,i,:,:,:]

            use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
            if i>0 and use_teacher_forcing==False:
                image_batch = img_output.detach()
                # image_batch = G[:,i-1,:,:,:].detach()

            # encoder
            audio_z = self.audio_encoder(audio_batch)
            image_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4 = self.image_encoder(image_batch)
            # concatenate
            if self.img_encoder_type=='FCN':
                audio_z = audio_z.unsqueeze(-1).unsqueeze(-1)
                audio_z = audio_z.repeat(1,1, image_z.shape[2], image_z.shape[3])
            concat_z = torch.cat([image_z, audio_z], dim=1) #(batch_size, c, h, w)

            # rnn
            concat_z = concat_z.unsqueeze(0) #(1, batch_size, c, h, w)
            rnn_output, hidden = self.rnn(concat_z, hidden)

            # decoder
            rnn_output = rnn_output.squeeze(0) #(batch_size, c, h, w)
            img_output = self.image_decoder(rnn_output, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)

            if i==0:
                G = img_output
                G = G.unsqueeze(1)
            else:

                G = torch.cat([G, img_output.unsqueeze(1)], dim=1)
            # G[:,i,:,:,:] = img_output



        # for batch_id in range(batch_size):
        #     valid_l = valid_len[batch_id]
        #     image_seq = image_inputs[batch_id] # (seq_len, c, h, w)
        #     audio_seq = audio_inputs[batch_id] # (seq_len, ...)
        #     # encoder z
        #     audio_seq_z = self.audio_encoder(audio_seq)

        #     # fed concat_z to RNN
        #     hidden = self.rnn.init_hidden(batch_size=1)
        #     use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        #     if use_teacher_forcing: # Teacher forcing: Feed the target as the next input
        #         image_z,  img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4 = self.image_encoder(image_seq)
        #         audio_z = audio_seq_z
        #         if self.img_encoder_type=='FCN':
        #             audio_z = audio_z.unsqueeze(-1).unsqueeze(-1)
        #             audio_z = audio_z.repeat(1,1, image_z.shape[2], image_z.shape[3])
        #         concat_z = torch.cat([image_z, audio_z], dim=1)

        #         # rnn
        #         concat_z = concat_z.unsqueeze(1)
        #         rnn_output, _ = self.rnn(concat_z, hidden)

        #         # decoder
        #         rnn_output = rnn_output.squeeze(1)
        #         img_output = self.image_decoder(rnn_output, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)
        #         G[batch_id] = img_output

        #     else: # Without teacher forcing: use its own predictions as the next input
        #         image_input = image_seq[0].unsqueeze(0) # (1, c, h, w)
        #         for i in range(valid_l):
        #             image_z, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4 = self.image_encoder(image_input)
        #             audio_z = audio_seq_z[i].unsqueeze(0) # (1, ...)

        #             if self.img_encoder_type=='FCN':
        #                 audio_z = audio_z.unsqueeze(-1).unsqueeze(-1)
        #                 audio_z = audio_z.repeat(1,1, image_z.shape[2], image_z.shape[3])
        #             concat_z = torch.cat([image_z, audio_z], dim=1)

        #             # rnn
        #             concat_z = concat_z.unsqueeze(1)
        #             rnn_output, hidden = self.rnn(concat_z, hidden)

        #             # decoder
        #             rnn_output = rnn_output.squeeze(1)
        #             img_output = self.image_decoder(rnn_output, img_e_conv1, img_e_conv2, img_e_conv3, img_e_conv4)

        #             image_input = img_output.detach()
        #             G[batch_id, i] = img_output

        return G

    def model_type(self):
        return 'RNN'






