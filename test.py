import os, argparse, collections, pprint, glob, sys
import traceback
import numpy as np
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"]= '0'
import torch
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import *
# from dataloader import Resizer, Normalizer, ToTensor
import model_G as model_G
from utils import utils, data

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_parser(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--num_input_imgs', help='num of input images', type=int, default=1)
    parser.add_argument('--size_image', help='input image size', type=int, default=112)
    parser.add_argument('--audio_encoder', help='audio encoder network', type=str, default='reduce')
    parser.add_argument('--img_encoder', help='image encoder network', type=str, default='reduce')
    parser.add_argument('--img_decoder', help='image decoder network', type=str, default='reduce')
    parser.add_argument('--rnn_type', help='type or RNN', type=str, default=None)
    parser.add_argument('--discriminator', help='multi patch or cond', type=str, default=None)
    parser.add_argument('--ckpt', help='pretrained model folder', type=str, default=None)
    parser.add_argument('--save_dir', help='saving folder path', type=str, default='save')
    parser.add_argument('--test_dir', help='testing sample path', type=str, default=None)
    parser.add_argument('--num_output_length', help='the length of feature', type=int, default=50)

    parser.add_argument('--is_train', help='if train', type=bool, default=False)
    parser.add_argument('--is_region', help='if train with only mouth region', type=bool, default=False)
    parser.add_argument('--if_tanh', help='if use tanh', type=bool, default=False)

    parser.add_argument('--gpu', help='which gpu to use', type=str, default='0' )

    config = parser.parse_args(args)
    config.gpu = config.gpu.split(',')
    print("Using gpu: ", config.gpu)
    return config

def test_cnn(config):
    num_gpu = len(config.gpu)

    G_model = model_G.LipGeneratorCNN(config.audio_encoder, config.img_encoder, config.img_decoder, config.size_image, config.num_output_length, config.if_tanh)
    # G_model.load_state_dict(torch.load(config.ckpt))
    load_ckpt(G_model, config.ckpt)


    if num_gpu>1:
        G_model = torch.nn.DataParallel(G_model, device_ids=list(range(num_gpu))).cuda()
    else:
        G_model = G_model.cuda()

    test(G_model, config.test_dir, config.save_dir, config.size_image)


def test_rnn(config):
    num_gpu = len(config.gpu)

    G_model = model_G.LipGeneratorRNN(config.audio_encoder, config.img_encoder, config.img_decoder, config.rnn_type,
                                    config.size_image, config.num_output_length, if_tanh = config.if_tanh)
    load_ckpt(G_model, config.ckpt)

    if num_gpu>1:
        G_model = torch.nn.DataParallel(G_model, device_ids=list(range(num_gpu))).cuda()
    else:
        G_model = G_model.cuda()

    test(G_model, config.test_dir, config.save_dir, config.size_image)




def test(model, test_dir, save_dir, image_size):
    model.eval()
    test_dirs = utils.listdir_nohidden(test_dir)
    for sub_folder in test_dirs:
        save_test_dir = os.path.join(save_dir, os.path.basename(sub_folder))

        audio_feature_files = glob.glob(os.path.join(sub_folder,'audio_sample/*.mat'))
        audio_feature_files = utils.sort_filename(audio_feature_files)
        image_test_file = os.path.join(sub_folder, 'image_sample.jpg')
        audio_test_file = os.path.join(sub_folder, 'audio_sample.wav')
        audio_duration = utils.get_wav_duration(audio_test_file)

        input_image = data.load_image(image_test_file, image_size)
        input_audios = [data.load_audio(audio_feature_file) for audio_feature_file in audio_feature_files]
        input_images = [input_image]*len(input_audios)

        # convert to tensor
        input_images = torch.from_numpy(np.array(input_images).transpose((0, 3, 1, 2))).cuda()
        input_audios = torch.from_numpy(np.array(input_audios).transpose((0, 3, 1, 2))).cuda()

        print("input image shape: ", input_images.size())
        print("input audio shape: ", input_audios.size())

        model_type =  model.module.model_type() if isinstance(model, torch.nn.DataParallel) else model.model_type()

        if model_type=='RNN':
            input_images = input_images.unsqueeze(0)
            input_audios = input_audios.unsqueeze(0)
            G_images = model(input_images, input_audios, torch.tensor([input_audios.shape[1]], dtype=torch.int32).cuda())
            G_images = G_images.squeeze(0)
        else:
            G_images = model(input_images, input_audios)
        G_images = G_images.cpu().detach().numpy()

        utils.save_video(audio_duration, audio_test_file, G_images, save_test_dir)



def load_ckpt(model, ckpt_path):
    old_state_dict = torch.load(ckpt_path)
    cur_state_dict = model.state_dict()
    for param in cur_state_dict:
        if param in old_state_dict and cur_state_dict[param].size()==old_state_dict[param].size():
            print("loading param: ", param)
            model.state_dict()[param].data.copy_(old_state_dict[param].data)


if __name__ == '__main__':
    config = get_parser()
    utils.create_dir(config.save_dir)
    if config.rnn_type==None:
        test_cnn(config)
    else:
        test_rnn(config)

