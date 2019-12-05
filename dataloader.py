from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

# import skimage.io
# import skimage.transform
# import skimage.color
# import skimage
from scipy.misc import imread, imresize, imsave

from scipy.io import loadmat
# from ffmpy import FFmpeg

def load_image(img_path):
    img = imread(img_path)
    if len(img.shape)==2:
        img = np.stack((img,)*3, axis=0)
    if len(img.shape)!=3:
        print("***** not rgb image ******", img_path, img.shape)
    return img.astype(np.float32)


def load_audio(audio_path):
    # load the mat file
    file = loadmat(audio_path)
    audio = np.transpose(file['mfcc'],(1,0))[1:,:]
    audio = np.expand_dims(audio, axis=-1).astype(np.float32)
    return audio


def load_mask(mask_path):
    img = imread(mask_path, flatten=True)/255.0
    return  img.astype(np.float32)


def read_csv_file(path, use_mask, num_input_imgs, num_gt_imgs, use_lip=False, use_word_label=False):
    filename = os.path.abspath(path)
    input_imgs, gt_imgs, input_audios, lip_coordinates, word_labels = [], [], [], [], []
    with open(filename) as f:
        lines = f.readlines()
    for line in lines:
        elems = line.rstrip('\n').split(' ')
        # if len(elems) < num_input_imgs + num_gt_imgs + 1:
        #     raise ValueError("input file list has less elements as needed")
        input_imgs.append(elems[:num_input_imgs])
        gt_imgs.append(elems[num_input_imgs : num_input_imgs+num_gt_imgs])
        input_audios.append(elems[num_input_imgs+num_gt_imgs : num_input_imgs+num_gt_imgs+1])

        if use_lip==True:
            x1, y1, x2, y2 = elems[num_input_imgs+num_gt_imgs+1].split(',')
            lip_coordinates.append( list(map(int, [x1, y1, x2, y2])))
        if use_word_label:
            word_labels.append(int(elems[-1]))


    return input_imgs, gt_imgs, input_audios, lip_coordinates, word_labels


class SeqDataset(Dataset):
    def __init__(self, train_file, use_mask, config, transform=None):
        self.train_file = train_file
        self.use_mask = use_mask
        self.num_gt_imgs= config.num_gt_imgs
        self.num_input_imgs = config.num_input_imgs
        self.num_seq_length = config.num_seq_length
        self.use_lip = config.use_lip
        self.use_word_label = config.use_word_label
        self.transform = transform

        self.input_img_seq = []
        self.gt_seq = []
        self.input_audio_seq = []
        self.lip_coord_seq = []
        self.word_label_seq = []
        image_list, gt_list, audio_list, coord_list, word_label_list = read_csv_file(train_file, use_mask, self.num_input_imgs, self.num_gt_imgs, self.use_lip, self.use_word_label)
        self.load_seq_input(image_list, gt_list, audio_list, coord_list, word_label_list)
        print("total number of sequence: ", len(self.input_img_seq))


    def load_seq_input(self, image_list, gt_list, audio_list, coord_list, word_label_list):
        seq_dic = defaultdict(list)
        coord_dic = defaultdict(list)
        word_dic = defaultdict(list)

        for i in range(len(image_list)):
            basepath, filename = os.path.split(image_list[i][0])
            celebname = os.path.basename(basepath)
            audioname = '_'.join(filename.split('_')[:-1])
            key = celebname +'/'+ audioname

            if self.use_lip and self.use_word_label:
                seq_dic[key].append([image_list[i], gt_list[i], audio_list[i], coord_list[i], word_label_list[i]])
            elif self.use_lip:
                seq_dic[key].append([image_list[i], gt_list[i], audio_list[i], coord_list[i]])
            elif self.use_word_label:
                seq_dic[key].append([image_list[i], gt_list[i], audio_list[i], word_label_list[i]])
            else:
                seq_dic[key].append([image_list[i], gt_list[i], audio_list[i]])

        for key, items in seq_dic.items():
            self.input_img_seq.append([elem[0] for elem in items])
            self.gt_seq.append([elem[1] for elem in items])
            self.input_audio_seq.append([elem[2] for elem in items])
            if self.use_lip:
                self.lip_coord_seq.append([elem[3] for elem in items])
            if self.use_word_label:
                self.word_label_seq.append(items[0][-1])

        print("total sequence: ", len(self.input_img_seq))


    def __len__(self):
        return len(self.input_img_seq)

    def __getitem__(self, idx):
        images, gts, audios, lips = [], [], [], []


        if len(self.input_img_seq[idx]) > self.num_seq_length:
            start_i = random.randint(0, len(self.input_img_seq[idx]) - self.num_seq_length)
            end_i = min(len(self.input_img_seq[idx]), start_i+self.num_seq_length)
        else:
            start_i, end_i = 0, len(self.input_img_seq[idx])

        for i in range(start_i, end_i):
            images.append([load_image(img_path) for img_path in self.input_img_seq[idx][i]])  #(seq_len, num_input_imgs, h,w,c)
            gts.append([load_image(img_path) for img_path in self.gt_seq[idx][i]])
            audios.append([load_audio(audio_path) for audio_path in self.input_audio_seq[idx][i]])

            if self.use_lip:
                lips.append(self.lip_coord_seq[idx][i]) #(seq_len, 4)
            if self.use_word_label:
                label = self.word_label_seq[idx] # int

        if self.use_word_label and self.use_lip:
            sample = {'img': images, 'audio': audios, 'gt': gts, 'lip': lips, 'label': label}
        elif self.use_lip:
            sample = {'img': images, 'audio': audios, 'gt': gts, 'lip': lips}
        elif self.use_word_label:
            sample = {'img': images, 'audio': audios, 'gt': gts, 'label': label}
        else:
            sample = {'img': images, 'audio': audios, 'gt': gts}
        if self.transform:
            sample = self.transform(sample)

        return sample


class CSVDataset(Dataset):
    """CSV dataset."""

    def __init__(self, train_file, use_mask, num_input_imgs, num_gt_imgs, transform=None):
        """
        Args:
            train_file (string): CSV file with training annotations
            annotations (string): CSV file with class list
            test_file (string, optional): CSV file with testing annotations
        """
        self.train_file = train_file
        self.use_mask = use_mask
        self.num_gt_imgs= num_gt_imgs
        self.num_input_imgs = num_input_imgs
        self.transform = transform

        self.input_imgs, self.gt_imgs, self.input_audios, _, _ = read_csv_file(train_file, use_mask, num_input_imgs, num_gt_imgs)


    def __len__(self):
        return len(self.input_imgs)

    def __getitem__(self, idx):
        images = [load_image(img_path) for img_path in self.input_imgs[idx]]
        gts = [load_image(img_path) for img_path in self.gt_imgs[idx]]
        audios = [load_audio(audio_path) for audio_path in self.input_audios[idx]]
        # audios = load_audio(self.input_audios[idx][0])


        sample = {'img': images, 'audio': audios, 'gt': gts}
        if self.transform:
            sample = self.transform(sample)
        return sample


# class NpySeqDataset(Dataset):

#     def __init__(self, train_file, config, transform=None):
#         self.transform = transform
#         self.img_seq = []
#         self.audio_seq = []
#         self.label_seq = []
#         self.lip_coord_seq = []
#         self.use_word_label = config.use_word_label
#         self.use_lip = config.use_lip
#         self.read_npy_file(train_file)
#         print('length of training list: ', len(self.img_seq))

#     def read_npy_file(self, train_file):
#         with open(train_file) as f:
#             lines = f.readlines()
#         for line in lines:
#             elems = line.rstrip('\n').split(' ')
#             self.img_seq.append(elems[0])
#             self.audio_seq.append(elems[1])
#             if self.use_lip:
#                 x1, y1, x2, y2 = elems[2].split(',')
#                 self.lip_coord_seq.append(list(map(int, [x1, y1, x2, y2])))

#             if self.use_word_label:
#                 self.label_seq.append(int(elems[-1]))

#     def __len__(self):
#         return len(self.img_seq)

#     def __getitem__(self, idx):
#         images = np.load(self.img_seq[idx]) # np array: (seq_len, h, w, 3)
#         audios = np.load(self.audio_seq[idx]) # np array: (seq_len, h, w, 1)
#         if self.use_word_label:
#             label = self.label_seq[idx] # int

#         if audios.shape[0]!=11 and audios.shape[1]!=12 and audios.shape[2]!=35 and audios.shape[3]!=1:
#             print("********** audio shape does not match ************", self.audio_seq[idx])
#             sys.exit()

#         input_images, gt_images, input_audios, lips = [], [], [], []

#         for i in range(images.shape[0]):
#             input_images.append([images[0]])
#             gt_images.append([images[i]])
#             input_audios.append([audios[i]])
#             if self.use_lip:
#                 lips.append(self.lip_coord_seq[idx])

#         if self.use_word_label and self.use_lip:
#             sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images, 'lip': lips, 'label': label}
#         elif self.use_word_label:
#             sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images, 'label': label}
#         elif self.use_lip:
#             sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images, 'lip': lips}
#         else:
#             sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images}

#         if self.transform:
#             sample = self.transform(sample)
#         return sample



class NpySeqDataset(Dataset):

    def __init__(self, train_file, config, transform=None):
        self.transform = transform
        self.img_seq = []
        self.gt_seq = []
        self.audio_seq = []
        self.label_seq = []
        self.lip_coord_seq = []
        self.use_word_label = config.use_word_label
        self.use_lip = config.use_lip
        self.read_npy_file(train_file)
        print('length of training list: ', len(self.img_seq))

    def read_npy_file(self, train_file):
        with open(train_file) as f:
            lines = f.readlines()
        for line in lines:
            elems = line.rstrip('\n').split(' ')
            self.img_seq.append(elems[0])
            self.gt_seq.append(elems[1])
            self.audio_seq.append(elems[2])
            if self.use_lip:
                coords = []
                for i in range(11):
                    x1, y1, x2, y2 = list(map(int,elems[3+i].split(',')))
                    if x2-x1>40:
                        print('mouth size > 40')
                        x2 -= (x2-x1)-40
                    if y2-y1>40:
                        print('mouth size > 40')
                        y2 -= (y2-y1)-40
                    if x2-x1<40:
                        print('mouth size < 40')
                        x2 += 40 - (x2-x1)
                    if y2-y1<40:
                        print('mouth size < 40')
                        y2 += 40 - (y2-y1)
                    coords.append([x1,y1,x2,y2]) # [[x1, y1, x2, y2],...[x1, y1, x2, y2]]
                self.lip_coord_seq.append(coords) # [coords, ..., coords]
            if self.use_word_label:
                self.label_seq.append(int(elems[-1]))

    def __len__(self):
        return len(self.img_seq)

    def __getitem__(self, idx):
        images = np.load(self.img_seq[idx]) # np array: (seq_len, h, w, 3)
        gts = np.load(self.gt_seq[idx])  # np array: (seq_len, h, w, 3)
        audios = np.load(self.audio_seq[idx]) # np array: (seq_len, h, w, 1)
        if self.use_word_label:
            label = self.label_seq[idx] # int

        if audios.shape[0]!=11 and audios.shape[1]!=12 and audios.shape[2]!=35 and audios.shape[3]!=1:
            print("********** audio shape does not match ************", self.audio_seq[idx])
            sys.exit()

        input_images, gt_images, input_audios, lips = [], [], [], []

        for i in range(images.shape[0]):
            input_images.append([images[i]])
            gt_images.append([gts[i]])
            input_audios.append([audios[i]])
        if self.use_lip:
            lips = self.lip_coord_seq[idx]  # [[x1, y1, x2, y2],...[x1, y1, x2, y2]]

        if self.use_word_label and self.use_lip:
            sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images, 'lip': lips, 'label': label}
        elif self.use_word_label:
            sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images, 'label': label}
        elif self.use_lip:
            sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images, 'lip': lips}
        else:
            sample = {'img': input_images, 'audio': input_audios, 'gt': gt_images}

        if self.transform:
            sample = self.transform(sample)
        return sample



def convert_seq_to_batch(data):
    img_batch = [s['img'] for s in data]  # [(seq_len, h, w, c), ..., (seq_len, h, w, c)]
    audio_batch = [s['audio'] for s in data]
    gt_batch = [s['gt'] for s in data]
    lip_batch, label_batch = None, None
    if 'lip' in data[0]:
        lip_batch = [s['lip'] for s in data]   # [(seq_len, 4), ..., (seq_len, 4)]
    if 'label' in data[0]:
        label_batch = [s['label'] for s in data]  # [label, ..., label]

    seq_len_batch = np.array([s.shape[0] for s in img_batch], dtype=np.int32) #[seq_len1, seq_len2, ...]


    # padding to the same length
    max_len = int(max(seq_len_batch))
    batch_size = len(img_batch)
    padded_img_batch = torch.zeros(batch_size, max_len, img_batch[0].shape[1], img_batch[0].shape[2], img_batch[0].shape[3])
    padded_gt_batch = torch.zeros(batch_size, max_len, gt_batch[0].shape[1], gt_batch[0].shape[2], gt_batch[0].shape[3])
    padded_audio_batch = torch.zeros(batch_size, max_len, audio_batch[0].shape[1], audio_batch[0].shape[2], audio_batch[0].shape[3])
    if lip_batch is not None:
        padded_lip_batch = torch.zeros((batch_size, max_len, lip_batch[0].shape[1]), dtype=torch.int32)  # (batch_size, max_len, 4)
    if label_batch is not None:
        label_batch = torch.stack(label_batch) # (batch_size,)

    for i in range(batch_size):
        padded_img_batch[i,:img_batch[i].shape[0],:,:,:] = img_batch[i]
        padded_gt_batch[i,:gt_batch[i].shape[0],:,:,:] = gt_batch[i]
        padded_audio_batch[i,:audio_batch[i].shape[0],:,:,:] = audio_batch[i]
        if lip_batch is not None:
            padded_lip_batch[i,:lip_batch[i].shape[0],:] = lip_batch[i]


    # reshape
    padded_img_batch = padded_img_batch.contiguous().view(batch_size*max_len, padded_img_batch.shape[2], padded_img_batch.shape[3], padded_img_batch.shape[4])
    padded_audio_batch = padded_audio_batch.contiguous().view(batch_size*max_len, padded_audio_batch.shape[2], padded_audio_batch.shape[3], padded_audio_batch.shape[4])
    padded_gt_batch = padded_gt_batch.contiguous().view(batch_size*max_len, padded_gt_batch.shape[2], padded_gt_batch.shape[3], padded_gt_batch.shape[4])
    if lip_batch is not None:
        padded_lip_batch = padded_lip_batch.contiguous().view(batch_size*max_len, padded_lip_batch.shape[2])


    if lip_batch is not None and label_batch is not None:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len_batch), 'label': label_batch, 'lip': padded_lip_batch}
    elif lip_batch is not None:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len_batch), 'lip': padded_lip_batch}
    elif label_batch is not None:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len_batch), 'label': label_batch}
    else:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len_batch)}




def collater(data):
    img_batch = [s['img'] for s in data]
    audio_batch = [s['audio'] for s in data]
    gt_batch = [s['gt'] for s in data]
    lip_batch = None
    label_batch = None
    if 'lip' in data[0]:
        lip_batch = [s['lip'] for s in data]   # (batch_size, seq_len, 4)
    if 'label' in data[0]:
        label_batch = [s['label'] for s in data]  # [label, ..., label]

    img_batch = sorted(img_batch, key=lambda x: x.shape[0], reverse=True)
    audio_batch = sorted(audio_batch, key=lambda x: x.shape[0], reverse=True)
    gt_batch = sorted(gt_batch, key=lambda x: x.shape[0], reverse=True)
    if lip_batch is not None:
        lip_batch = sorted(lip_batch,  key=lambda x: x.shape[0], reverse=True)

    seq_len = np.array([s.shape[0] for s in img_batch], dtype=np.int32)

    # padding to max_len
    max_len = img_batch[0].shape[0]
    batch_size = len(img_batch)

    padded_img_batch = torch.zeros(batch_size, max_len, img_batch[0].shape[1], img_batch[0].shape[2], img_batch[0].shape[3])
    padded_gt_batch = torch.zeros(batch_size, max_len, gt_batch[0].shape[1], gt_batch[0].shape[2], gt_batch[0].shape[3])
    padded_audio_batch = torch.zeros(batch_size, max_len, audio_batch[0].shape[1], audio_batch[0].shape[2], audio_batch[0].shape[3])
    if lip_batch is not None:
        padded_lip_batch = torch.zeros((batch_size, max_len, lip_batch[0].shape[1]), dtype=torch.int32)  # (batch_size, max_len, 4)
    if label_batch is not None:
        label_batch = torch.stack(label_batch) # (batch_size,)

    for i in range(batch_size):
        padded_img_batch[i,:img_batch[i].shape[0],:,:,:] = img_batch[i]
        padded_gt_batch[i,:gt_batch[i].shape[0],:,:,:] = gt_batch[i]
        padded_audio_batch[i,:audio_batch[i].shape[0],:,:,:] = audio_batch[i]
        if lip_batch is not None:
            padded_lip_batch[i,:lip_batch[i].shape[0],:] = lip_batch[i]

    if lip_batch is not None and label_batch is not None:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len), 'label': label_batch, 'lip': padded_lip_batch}
    elif lip_batch is not None:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len), 'lip': padded_lip_batch}
    elif label_batch is not None:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len), 'label': label_batch}
    else:
        return {'img': padded_img_batch, 'audio': padded_audio_batch, 'gt': padded_gt_batch, 'len': torch.from_numpy(seq_len)}



class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    def __init__(self, image_size=112):
        self.image_size = image_size

    def __call__(self, sample):
        images, gts = sample['img'], sample['gt']  # (seq_len, num_input_imgs, h,w,c) or (num_input_imgs, h,w,c)

        re_images, re_gts, re_lips = [], [], []
        for i in range(len(images)):
            if isinstance(images[i], list):
                re_images.append([imresize(img, [self.image_size, self.image_size]) for img in images[i]])
                re_gts.append([imresize(img, [self.image_size, self.image_size]) for img in gts[i]])
            else: #(h, w, c)
                re_images.append(imresize(images[i], [self.image_size, self.image_size]))
                re_gts.append(imresize(gts[i], [self.image_size, self.image_size]))

        sample['img'] = re_images
        sample['gt'] = re_gts

        return sample

class Normalizer(object):

    def __call__(self, sample):
        images, gts = sample['img'], sample['gt']
        norm_images, norm_gts = [], []
        for i in range(len(images)):
            if isinstance(images[i], list): # sequence input
                norm_images.append([(img*2.0/255.0-1.0).astype(np.float32) for img in images[i]])
                norm_gts.append([(gt*2.0/255.0-1.0).astype(np.float32) for gt in gts[i]])
            else:
                norm_images.append((images[i]*2.0/255.0-1.0).astype(np.float32))
                norm_gts.append((gts[i]*2.0/255.0-1.0).astype(np.float32))
        sample['img'] = norm_images
        sample['gt'] = norm_gts
        return sample


class ToTensor(object):
    def __call__(self, sample):
        images, audios, gts = sample['img'], sample['audio'], sample['gt']
        lips = sample['lip'] if 'lip' in sample else None
        label = sample['label'] if 'label' in sample else None

        if isinstance(images[0], list):# sequence input
            images = np.array([np.concatenate(imgs_, axis=-1).transpose((2, 0, 1)) for imgs_ in images])
            audios = np.array([np.concatenate(audios_, axis=-1).transpose((2, 0, 1)) for audios_ in audios])
            gts = np.array([np.concatenate(gts_, axis=-1).transpose((2, 0, 1)) for gts_ in gts])
        else:
            images = np.concatenate(images, axis=-1).transpose((2, 0, 1))
            audios = np.concatenate(audios, axis=-1).transpose((2, 0, 1))
            gts = np.concatenate(gts, axis=-1).transpose((2, 0, 1))

        if lips is not None and label is not None:
            return {'img': torch.from_numpy(images), 'audio': torch.from_numpy(audios), 'gt': torch.from_numpy(gts), 'lip': torch.LongTensor(lips), 'label': torch.tensor(label, dtype=torch.long)}
        elif lips is not None:
            return {'img': torch.from_numpy(images), 'audio': torch.from_numpy(audios), 'gt': torch.from_numpy(gts), 'lip': torch.LongTensor(lips)}
        elif label is not None:
            return {'img': torch.from_numpy(images), 'audio': torch.from_numpy(audios), 'gt': torch.from_numpy(gts),'label': torch.LongTensor(label)}
        else:
            return {'img': torch.from_numpy(images), 'audio': torch.from_numpy(audios), 'gt': torch.from_numpy(gts)}



class AspectRatioBasedSampler(Sampler):

    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]

