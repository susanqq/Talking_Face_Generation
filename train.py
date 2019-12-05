import os, argparse, collections, pprint, glob, sys
import traceback
import random
import numpy as np
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"]= '0'
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import *
import model_G as model_G
import model_D
import model_LipNet as LipNet
import loss
from utils import utils, data

print('CUDA available: {}'.format(torch.cuda.is_available()))

def get_parser(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=50)
    parser.add_argument('--batch_size', help='batch_size', type=int, default=4)
    parser.add_argument('--num_input_imgs', help='num of input images', type=int, default=1)
    parser.add_argument('--num_gt_imgs', help='num of ground truth images', type=int, default=1)
    parser.add_argument('--size_image', help='input image size', type=int, default=128)
    parser.add_argument('--audio_encoder', help='audio encoder network', type=str, default='reduce')
    parser.add_argument('--img_encoder', help='image encoder network', type=str, default='reduce')
    parser.add_argument('--img_decoder', help='image decoder network', type=str, default='reduce')
    parser.add_argument('--rnn_type', help='type of RNN: GRU', type=str, default=None)
    parser.add_argument('--discriminator', help='type of discriminator', type=str, default=None)
    parser.add_argument('--discriminator_lip', help='type of discriminator', type=str, default=None)
    parser.add_argument('--discriminator_v', help='type of discriminator', type=str, default=None)
    parser.add_argument('--D_v_weight', help='video discriminator weight', type=float, default=0.01)
    parser.add_argument('--D_lip_weight', help='lip read discriminator weight', type=float, default=0.001)
    parser.add_argument('--ckpt', help='pretrained model', type=str, default=None)
    parser.add_argument('--ckpt_lipmodel', help='pretrained lip read model', type=str, default=None)

    parser.add_argument('--save_dir', help='saving folder path', type=str, default='save')
    parser.add_argument('--test_dir', help='testing sample path', type=str, default=None)
    parser.add_argument('--filename_list', help='the training filename list', type=str, default=None)
    parser.add_argument('--teacher_force_ratio', help='ratio for using target image as input for rnn seq', type=float, default=0.5)
    parser.add_argument('--lr', help='learn rate', type=float, default=0.0002)
    parser.add_argument('--num_output_length', help='the length of feature', type=int, default=512)
    parser.add_argument('--num_seq_length', help='maximum length of input sequence', type=int, default=100)
    parser.add_argument('--num_frames_D', help='num of frames for video discriminator', type=int, default=20)
    parser.add_argument('--num_frames_lipNet', help='num of input frames for lip read', type=int, default=11)
    parser.add_argument('--if_vgg', help='if use vgg feature', type=bool, default=False)
    parser.add_argument('--use_npy', help='if use npy input', type=bool, default=False)
    parser.add_argument('--use_seq', help='if use seq input', type=bool, default=False)
    parser.add_argument('--use_word_label', help='if use word label', type=bool, default=False)
    parser.add_argument('--use_lip', help='if use lip coordinate', type=bool, default=False)
    parser.add_argument('--is_train', help='if train', type=bool, default=False)
    parser.add_argument('--is_region', help='if train with only mouth region', type=bool, default=False)
    parser.add_argument('--if_tanh', help='if use tanh', type=bool, default=False)

    parser.add_argument('--optimizer', help='the optimizer', type=str, default='ADAM')
    parser.add_argument('--gpu', help='which gpu to use', type=str, default='0' )

    config = parser.parse_args(args)
    config.gpu = config.gpu.split(',')
    print("Using gpu: ", config.gpu)
    return config

def train_cnn(config):
    num_gpu = len(config.gpu)
    if config.use_npy:
        dataset_train = NpySeqDataset(train_file = config.filename_list, config=config, transform = transforms.Compose([Resizer(config.size_image), Normalizer(), ToTensor()]))
        dataloader = DataLoader(dataset_train, num_workers=8,  pin_memory=False, collate_fn=convert_seq_to_batch, batch_size=config.batch_size, shuffle=True)
    elif config.use_seq:
        dataset_train = SeqDataset(train_file = config.filename_list,
                                    use_mask = None,
                                    config = config,
                                    transform = transforms.Compose([Resizer(config.size_image), Normalizer(), ToTensor()]))
        dataloader = DataLoader(dataset_train, num_workers=8, collate_fn=convert_seq_to_batch, batch_size=config.batch_size, shuffle=True)
    else:
        dataset_train = CSVDataset(train_file = config.filename_list,
                                use_mask = None,
                                num_input_imgs = config.num_input_imgs,
                                num_gt_imgs = config.num_gt_imgs,
                                transform = transforms.Compose([Resizer(config.size_image), Normalizer(), ToTensor()]))

        dataloader = DataLoader(dataset_train, num_workers=8,  pin_memory=False, batch_size=config.batch_size, shuffle=True)

    G_model = model_G.LipGeneratorCNN(config.audio_encoder, config.img_encoder, config.img_decoder, config.size_image, config.num_output_length, config.if_tanh)
    D_model_lip = model_D.DiscriminatorLip(config)
    D_model = model_D.Discriminator(config)
    D_v_model = model_D.DiscriminatorVideo(config)


    adversarial_loss_lip = loss.GAN_LR_Loss()
    adversarial_loss = loss.GANLoss()
    recon_loss = loss.ReconLoss()


    if config.ckpt!=None:
        load_ckpt(G_model, config.ckpt)
    if config.discriminator_lip == 'lip_read':
        load_ckpt(D_model_lip, config.ckpt_lipmodel, prefix='discriminator.')

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, G_model.parameters()), lr=0.002, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G_model.parameters(), lr=config.lr, betas=(0.5, 0.999))
    if config.discriminator is not None:
        optimizer_D = optim.Adam(D_model.parameters(), lr=0.001, betas=(0.5, 0.999))
    if config.discriminator_lip is not None:
        optimizer_D_lip = optim.Adam(D_model_lip.parameters(), lr=0.001, betas=(0.5, 0.999))
    if config.discriminator_v is not None:
        optimizer_D_v = optim.Adam(D_v_model.parameters(), lr=0.001, betas=(0.5, 0.999))
        
        
    if num_gpu>1:
        G_model = torch.nn.DataParallel(G_model, device_ids=list(range(num_gpu))).cuda()
        D_model = torch.nn.DataParallel(D_model, device_ids=list(range(num_gpu))).cuda()
        D_v_model = torch.nn.DataParallel(D_v_model, device_ids=list(range(num_gpu))).cuda()
        D_model_lip = torch.nn.DataParallel(D_model_lip, device_ids=list(range(num_gpu))).cuda()
    else:
        G_model = G_model.cuda()
        D_model = D_model.cuda()
        D_v_model = D_v_model.cuda()
        D_model_lip = D_model_lip.cuda()

    adversarial_loss = adversarial_loss.cuda()
    adversarial_loss_lip = adversarial_loss_lip.cuda()
    recon_loss = recon_loss.cuda()


    writer = SummaryWriter(log_dir=config.save_dir)

    sample_inputs = None
    for epoch_num in range(config.epochs):
        G_model.train()
        D_model.train()
        D_v_model.train()
        D_model_lip.train()
        for iter_num, data in enumerate(dataloader):
            n_iter = len(dataloader) * epoch_num + iter_num
            if sample_inputs==None:
                sample_inputs = (data['img'].cuda(), data['audio'].cuda(), data['gt'].cuda())
            try:
                input_images = data['img'].cuda()
                input_audios = data['audio'].cuda()
                gts = data['gt'].cuda()

                G_images = G_model(input_images, input_audios)

                if config.use_seq or config.use_npy:
                    # G_images = G_images.contiguous().view(config.batch_size, -1, G_images.shape[1], G_images.shape[2], G_images.shape[3])
                    loss_EG = recon_loss( G_images.contiguous().view(config.batch_size, -1, G_images.shape[1], G_images.shape[2], G_images.shape[3]),
                                          gts.contiguous().view(config.batch_size, -1, gts.shape[1], gts.shape[2], gts.shape[3]),
                                          valid_len=data['len'].cuda())
                else:
                    loss_EG = recon_loss(G_images, gts)
                G_loss = loss_EG

                if config.discriminator is not None:
                    loss_G_GAN = adversarial_loss(D_model(G_images), is_real=True)
                    loss_D_real = adversarial_loss(D_model(gts), is_real=True)
                    loss_D_fake = adversarial_loss(D_model(G_images.detach()), is_real=False)
                    G_loss = G_loss + 0.002*loss_G_GAN
                    D_loss = loss_D_real + loss_D_fake
                    
                if config.discriminator_v is not None:
                    # reshape G_images
                    G_images = G_images.contiguous().view(config.batch_size, -1, G_images.shape[1], G_images.shape[2], G_images.shape[3])
                    gts = gts.contiguous().view(config.batch_size, -1, gts.shape[1], gts.shape[2], gts.shape[3])
                    
                    clip_range = get_clip_range(data['len'], config.num_frames_D)
                    loss_G_GAN_v = adversarial_loss(D_v_model(G_images, config.num_frames_D, clip_range.cuda()), is_real=True)
                    loss_D_real_v = adversarial_loss(D_v_model(gts, config.num_frames_D, clip_range.cuda()), is_real=True)
                    loss_D_fake_v = adversarial_loss( D_v_model(G_images.detach(), config.num_frames_D, clip_range.cuda()), is_real=False)
                    G_loss = G_loss + config.D_v_weight*loss_G_GAN_v
                    D_loss_v = loss_D_real_v + loss_D_fake_v
                    
                if config.discriminator_lip is not None:
                    # reshape G_images
                    if len(G_images.shape)!=5:
                        G_images = G_images.contiguous().view(config.batch_size, -1, G_images.shape[1], G_images.shape[2], G_images.shape[3])
                        gts = gts.contiguous().view(config.batch_size, -1, gts.shape[1], gts.shape[2], gts.shape[3])
                    lip_coord =  data['lip'].contiguous().view(config.batch_size, -1, 4)

                    clip_range = get_clip_range(data['len'], config.num_frames_lipNet)
                
                    loss_G_GAN_lip = adversarial_loss_lip(D_model_lip(G_images, clip_range.cuda(), lip_coord.cuda()), is_real=True, targets=data['label'].cuda())
                    loss_D_real_lip = adversarial_loss_lip(D_model_lip(gts, clip_range.cuda(), lip_coord.cuda()), is_real=True, targets=data['label'].cuda())
                    loss_D_fake_lip = adversarial_loss_lip(D_model_lip(G_images.detach(), clip_range.cuda(), lip_coord.cuda()), is_real=False, targets=data['label'].cuda())

                    G_loss = G_loss + config.D_lip_weight*loss_G_GAN_lip
                    D_loss_lip = loss_D_real_lip + loss_D_fake_lip


                # for generator
                optimizer_G.zero_grad()
                G_loss.backward()
                optimizer_G.step()

                # for discriminator
                if config.discriminator is not None:
                    optimizer_D.zero_grad()
                    D_loss.backward()
                    optimizer_D.step()

                if config.discriminator_lip is not None:
                    optimizer_D_lip.zero_grad()
                    D_loss_lip.backward()
                    optimizer_D_lip.step()

                if iter_num%200==0:
                    test(G_model, config.test_dir, config.save_dir, config.size_image)
                    
                if iter_num % 20 == 0:    
                    print ('Epoch: {} | Iteration: {} | EG loss: {:1.5f}: '.format(epoch_num, iter_num,  float(loss_EG)))
                    if config.discriminator is not None:
                        print ('D loss {:1.5f} | G_GAN loss {:1.5f} : '.format(float(D_loss), float(loss_G_GAN)))
                        writer.add_scalar('D_loss', D_loss, n_iter)
                    if config.discriminator_v is not None:
                        print ('D_v loss: {:1.5f} | G_GAN_v loss {:1.5f} : '.format(float(D_loss_v), float(loss_G_GAN_v)))
                        writer.add_scalar('D_loss_v', D_loss_v, n_iter)
                    if config.discriminator_lip is not None:
                        print ('D_lip loss {:1.5f} | G_GAN_lip loss {:1.5f} : '.format(float(D_loss_lip), float(loss_G_GAN_lip)))
                        writer.add_scalar('D_loss_lip', D_loss_lip, n_iter)
                    writer.add_scalar('EG_loss', loss_EG, n_iter)
            except Exception as e:
                print(e)
                traceback.print_exc()

        # visualize some results
        sample(sample_inputs, G_model, epoch_num, config.save_dir)
        test(G_model, config.test_dir, config.save_dir, config.size_image)

        if isinstance(G_model, torch.nn.DataParallel):
            torch.save(G_model.module.state_dict(), os.path.join(config.save_dir, 'model_G{}.pt'.format(epoch_num)))
            if config.discriminator_lip is not None:
                torch.save(D_model_lip.module.state_dict(), os.path.join(config.save_dir, 'model_D{}.pt'.format(epoch_num)))
        else:
            torch.save(G_model.state_dict(), os.path.join(config.save_dir, 'model_G{}.pt'.format(epoch_num)))
            if config.discriminator_lip is not None:
                torch.save(D_model_lip.state_dict(), os.path.join(config.save_dir, 'model_D{}.pt'.format(epoch_num)))


    if isinstance(G_model, torch.nn.DataParallel):
        torch.save(G_model.module.state_dict(), os.path.join(config.save_dir, 'model_G_final.pt'.format(epoch_num)))
    else:
        torch.save(G_model.state_dict(), os.path.join(config.save_dir, 'model_G_final.pt'.format(epoch_num)))




def train_rnn(config):
    num_gpu = len(config.gpu)
    if config.use_npy:
        dataset_train = NpySeqDataset(train_file = config.filename_list, config=config, transform = transforms.Compose([Resizer(config.size_image), Normalizer(), ToTensor()]))
    elif config.use_seq:
        dataset_train = SeqDataset(train_file = config.filename_list,
                                    use_mask = None,
                                    config = config,
                                    transform = transforms.Compose([Resizer(config.size_image), Normalizer(), ToTensor()]))
    dataloader = DataLoader(dataset_train, num_workers=8, collate_fn=collater, batch_size=config.batch_size, shuffle=True)

    G_model = model_G.LipGeneratorRNN(config.audio_encoder, config.img_encoder, config.img_decoder, config.rnn_type,
                                    config.size_image, config.num_output_length, if_tanh = config.if_tanh)
    D_model = model_D.Discriminator(config)
    D_v_model = model_D.DiscriminatorVideo(config)
    D_model_lip = model_D.DiscriminatorLip(config)



    adversarial_lip_loss = loss.GAN_LR_Loss()
    adversarial_loss = loss.GANLoss()
    recon_loss = loss.ReconLoss()


    if config.ckpt is not None:
        load_ckpt(G_model, config.ckpt)
    if config.discriminator_lip == 'lip_read':
        load_ckpt(D_model_lip, config.ckpt_lipmodel,prefix='discriminator.')

    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, G_model.parameters()), lr=0.002, betas=(0.5, 0.999))
    optimizer_G = optim.Adam(G_model.parameters(), lr=config.lr, betas=(0.5, 0.999))
    if config.discriminator is not None:
        optimizer_D = optim.Adam(D_model.parameters(), lr=0.001, betas=(0.5, 0.999))
    if config.discriminator_v is not None:
        optimizer_D_v = optim.Adam(D_v_model.parameters(), lr=0.001, betas=(0.5, 0.999))
    if config.discriminator_lip is not None:
        optimizer_D_lip = optim.Adam(D_model_lip.parameters(), lr=0.001, betas=(0.5, 0.999))


    if num_gpu>1:
        G_model = torch.nn.DataParallel(G_model, device_ids=list(range(num_gpu))).cuda()
        D_model = torch.nn.DataParallel(D_model, device_ids=list(range(num_gpu))).cuda()
        D_v_model = torch.nn.DataParallel(D_v_model, device_ids=list(range(num_gpu))).cuda()
        D_model_lip = torch.nn.DataParallel(D_model_lip, device_ids=list(range(num_gpu))).cuda()
    else:
        G_model = G_model.cuda()
        D_model = D_model.cuda()
        D_model_lip = D_model_lip.cuda()
        D_v_model = D_v_model.cuda()

    adversarial_lip_loss = adversarial_lip_loss.cuda()
    adversarial_loss = adversarial_loss.cuda()
    recon_loss = recon_loss.cuda()



    writer = SummaryWriter(log_dir=config.save_dir)

    sample_inputs = None
    for epoch_num in range(config.epochs):
        G_model.train()
        D_model.train()
        D_v_model.train()
        D_model_lip.train()
        for iter_num, data in enumerate(dataloader):
            n_iter = len(dataloader) * epoch_num + iter_num
            if sample_inputs==None:
                sample_inputs = (data['img'].cuda(), data['audio'].cuda(), data['gt'].cuda(), data['len'].cuda())
            try:
                input_images = data['img'].cuda()
                input_audios = data['audio'].cuda()
                gts_orignal = data['gt'].cuda()

                G_images_orignal = G_model(input_images, input_audios,
                                    valid_len=data['len'].cuda(),
                                    teacher_forcing_ratio=config.teacher_force_ratio)
                loss_EG = recon_loss(G_images_orignal, gts_orignal, valid_len=data['len'].cuda())
                G_loss = loss_EG

                if config.discriminator is not None:
                    loss_G_GAN = adversarial_loss(D_model(G_images_orignal), is_real=True)
                    loss_D_real = adversarial_loss(D_model(gts_orignal), is_real=True)
                    loss_D_fake = adversarial_loss(D_model(G_images_orignal.detach()), is_real=False)
                    G_loss = G_loss + 0.002*loss_G_GAN
                    D_loss = loss_D_real + loss_D_fake


                if config.discriminator_v is not None:
                    clip_range = get_clip_range(data['len'], config.num_frames_D)
                    loss_G_GAN_v = adversarial_loss(D_v_model(G_images_orignal, config.num_frames_D, clip_range.cuda()), is_real=True)
                    loss_D_real_v = adversarial_loss(D_v_model(gts_orignal, config.num_frames_D, clip_range.cuda()), is_real=True)
                    loss_D_fake_v = adversarial_loss( D_v_model(G_images_orignal.detach(), config.num_frames_D, clip_range.cuda()), is_real=False)
                    G_loss = G_loss + config.D_v_weight*loss_G_GAN_v
                    D_loss_v = loss_D_real_v + loss_D_fake_v


                if config.discriminator_lip is not None:
                    clip_range = get_clip_range(data['len'], config.num_frames_lipNet)
                    
                    loss_G_GAN_lip = adversarial_lip_loss(D_model_lip(G_images_orignal, clip_range.cuda(), data['lip'].cuda()), is_real=True, targets=data['label'].cuda())
                    loss_D_real_lip = adversarial_lip_loss(D_model_lip(gts_orignal, clip_range.cuda(), data['lip'].cuda()), is_real=True, targets=data['label'].cuda())
                    loss_D_fake_lip = adversarial_lip_loss(D_model_lip(G_images_orignal.detach(), clip_range.cuda(), data['lip'].cuda()), is_real=False, targets=data['label'].cuda())
                    G_loss = G_loss + config.D_lip_weight*loss_G_GAN_lip
                    D_loss_lip = loss_D_real_lip + loss_D_fake_lip



                # for generator
                optimizer_G.zero_grad()
                G_loss.backward()
                optimizer_G.step()

                # for discriminator
                if config.discriminator is not None:
                    optimizer_D.zero_grad()
                    D_loss.backward()
                    optimizer_D.step()

                if config.discriminator_lip is not None:
                    optimizer_D_lip.zero_grad()
                    D_loss_lip.backward()
                    optimizer_D_lip.step()

                if config.discriminator_v is not None:
                    optimizer_D_v.zero_grad()
                    D_loss_v.backward()
                    optimizer_D_v.step()


                if iter_num % 20 == 0:
                    print ('Epoch: {} | Iteration: {} | EG loss: {:1.5f}: '.format(epoch_num, iter_num,  float(loss_EG)))
                    if config.discriminator is not None:
                        print ('D loss {:1.5f} | G_GAN loss {:1.5f} : '.format(float(D_loss), float(loss_G_GAN)))
                        writer.add_scalar('D_loss', D_loss, n_iter)
                    if config.discriminator_v is not None:
                        print ('D_v loss: {:1.5f} | G_GAN_v loss {:1.5f} : '.format(float(D_loss_v), float(loss_G_GAN_v)))
                        writer.add_scalar('D_loss_v', D_loss_v, n_iter)
                    if config.discriminator_lip is not None:
                        print ('D_lip loss {:1.5f} | G_GAN_lip loss {:1.5f} : '.format(float(D_loss_lip), float(loss_G_GAN_lip)))
                        writer.add_scalar('D_loss_lip', D_loss_lip, n_iter)

                    writer.add_scalar('loss_EG', loss_EG, n_iter)

            except Exception as e:
                print(e)
                traceback.print_exc()

        # visualize some results
        sample(sample_inputs, G_model, epoch_num, config.save_dir, config.teacher_force_ratio)
        test(G_model, config.test_dir, config.save_dir, config.size_image)


        if isinstance(G_model, torch.nn.DataParallel):
            torch.save(G_model.module.state_dict(), os.path.join(config.save_dir, 'model_G{}.pt'.format(epoch_num)))
            if config.discriminator_lip is not None:
                torch.save(D_model_lip.module.state_dict(), os.path.join(config.save_dir, 'model_D{}.pt'.format(epoch_num)))
        else:
            torch.save(G_model.state_dict(), os.path.join(config.save_dir, 'model_G{}.pt'.format(epoch_num)))
            if config.discriminator_lip is not None:
                torch.save(D_model_lip.state_dict(), os.path.join(config.save_dir, 'model_D{}.pt'.format(epoch_num)))

    if isinstance(G_model, torch.nn.DataParallel):
        torch.save(G_model.module.state_dict(), os.path.join(config.save_dir, 'model_G_final.pt'))
    else:
        torch.save(G_model.state_dict(), os.path.join(config.save_dir, 'model_G_final.pt'))


def get_clip_range(valid_len, required_len):
    clip_range = []
    for seq_len in valid_len.numpy():
        if seq_len > required_len:
            start_i = random.randint(0, seq_len-required_len)
            end_i = start_i + required_len
        else:
            start_i = 0
            end_i = seq_len
        clip_range.append([start_i, end_i])
    return torch.LongTensor(clip_range)






def sample(sample_inputs, model, epoch, save_dir, teacher_forcing_ratio=0):
    model.eval()
    sample_dir = os.path.join(save_dir, 'sample')
    utils.create_dir(sample_dir)

    model_type =  model.module.model_type() if isinstance(model, torch.nn.DataParallel) else model.model_type()

    input_images, input_audios, gt_images = sample_inputs[0], sample_inputs[1], sample_inputs[2]
    with torch.no_grad():
        if model_type=='RNN':
            G_images = model(input_images, input_audios,
                             valid_len = sample_inputs[3],
                             teacher_forcing_ratio = teacher_forcing_ratio)
            G_images = G_images.contiguous().view(G_images.shape[0]*G_images.shape[1],G_images.shape[2], G_images.shape[3], G_images.shape[4])
            input_images = input_images.contiguous().view(input_images.shape[0]*input_images.shape[1],input_images.shape[2], input_images.shape[3], input_images.shape[4])
            gt_images = gt_images.contiguous().view(gt_images.shape[0]*gt_images.shape[1],gt_images.shape[2], gt_images.shape[3], gt_images.shape[4])

        else:
            G_images = model(input_images, input_audios)

    # save input images
    utils.save_sample_images(input_images.cpu().detach().numpy(), os.path.join(sample_dir, 'input.png'))
    # save ground truth images
    utils.save_sample_images(gt_images.cpu().detach().numpy(), os.path.join(sample_dir, 'ground_truth.png'))
    # save generated images
    g_name = '{:02d}.png'.format(epoch+1)
    utils.save_sample_images(G_images.cpu().detach().numpy(), os.path.join(sample_dir, g_name))
    model.train()

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
        input_images = torch.from_numpy(np.array(input_images).transpose((0, 3, 1, 2))).cuda() # (seq_len, c, h, w)
        input_audios = torch.from_numpy(np.array(input_audios).transpose((0, 3, 1, 2))).cuda()

        model_type =  model.module.model_type() if isinstance(model, torch.nn.DataParallel) else model.model_type()

        with torch.no_grad():
            if model_type=='RNN':
                input_images = input_images.unsqueeze(0) # (1, seq_len, c, h, w)
                input_audios = input_audios.unsqueeze(0)
                G_images = model(input_images, input_audios,
                                valid_len = torch.tensor([input_audios.shape[1]], dtype=torch.int32).cuda(),
                                teacher_forcing_ratio = 0)
                G_images = G_images.squeeze(0)
            else:
                G_images = model(input_images, input_audios)
        utils.save_video(audio_duration, audio_test_file, G_images.cpu().detach().numpy(), save_test_dir)
        model.train()


def load_ckpt(model, ckpt_path, prefix=None):
    old_state_dict = torch.load(ckpt_path)
    cur_state_dict = model.state_dict()
    for param in cur_state_dict:
        if prefix is not None:
            old_param = param.replace(prefix, '')
        else:
            old_param = param
        if old_param in old_state_dict and cur_state_dict[param].size()==old_state_dict[old_param].size():
            print("loading param: ", param)
            model.state_dict()[param].data.copy_(old_state_dict[old_param].data)
        else:
            print("warning cannot load param: ", param)


def freeze_model(model):
    # model.eval()
    for params in model.parameters():
        params.requires_grad = False
    if isinstance(model, torch.nn.DataParallel):
        model.module.freeze_bn()
    else:
        model.freeze_bn()





if __name__ == '__main__':
    config = get_parser()
    utils.create_dir(config.save_dir)

    with open(os.path.join(config.save_dir, 'config.txt'), 'w') as f:
        dic = vars(config)
        pp = pprint.PrettyPrinter(indent=1, width=80, depth=None, stream=f)
        pp.pprint(dic)

    if config.rnn_type==None:
        train_cnn(config)
    else:
        train_rnn(config)

