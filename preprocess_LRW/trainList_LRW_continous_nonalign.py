import os
import glob
import sys
import re
import random
from collections import defaultdict




dataset = 'train'
AUDIO_DIR = '/SLP_Extended/susan.s/lrw_data'
IMG_DIR = '/SLP_Extended_1_0/data/aivision/bbc_word/non_align'
audio_feature_dir = os.path.join(AUDIO_DIR, 'audio_features', dataset)
input_img_dir = os.path.join(IMG_DIR, 'input_image', dataset)
gt_img_dir = os.path.join(IMG_DIR, 'gt_image', dataset)

train_filename='train_list_LRW_350ms_non_align.txt'
len_range = [11, 20]
spliter = '_'


f_train = open(train_filename, 'w')



def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def listdir_nohidden(path):
  file_list = []
  for f in os.listdir(path):
    if not f.startswith('.'):
      file_list.append(f)
  return file_list

def get_training_list(file_list):
  sentence_list = ['_'.join(filename.split('/')[-1].split('_')[:-1]) for filename in file_list]
  sentence_list = list(set(sentence_list))
  random.shuffle(sentence_list)
  num_train = int(len(sentence_list)*ratio)
  return sentence_list[:num_train]


if __name__ == '__main__':
  train_list = listdir_nohidden(input_img_dir)
  train_list.sort()

  total_num = 0
  # for each speaker
  for sub_folder in train_list:
    print(sub_folder)

    image_list = glob.glob(os.path.join(input_img_dir, sub_folder) + '/*.jpg')

    image_list.sort(key = alphanum_key)


    dic = defaultdict(list)
    for img_path in image_list:
      filedir, filename = os.path.split(img_path)
      word = os.path.basename(filedir)
      videoname = '_'.join(filename.split(spliter)[:-1])
      key = word +'/'+ videoname
      dic[key].append(img_path)


    for key, item in dic.iteritems():
      if len(item)>=len_range[0] and len(item)<=len_range[1]:
        item.sort(key = alphanum_key)
        total_num += len(item)
        for input_img in item:
          gt_img = os.path.join(gt_img_dir, sub_folder) + '/' + os.path.basename(input_img)
          input_audio = os.path.join(audio_feature_dir, sub_folder) + '/' +os.path.basename(input_img).replace('jpg', 'mat')
          f_train.write(input_img + ' ' + gt_img + ' ' + input_audio + '\n')

      print(key, len(item))


  print(total_num)

  f_train.close()
  # f_test.close()


