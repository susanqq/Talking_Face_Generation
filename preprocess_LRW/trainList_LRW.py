import os
import glob
import sys
import re
import random





dataset = 'test'
feature_dir='/home/ysong18/Documents/Speech2Vid/data/LRW_0.35'
audio_feature_dir = os.path.join(feature_dir, 'audio_feature', dataset)
img_feature_dir = os.path.join(feature_dir, 'extract_faces', dataset)
train_filename='train_list_test_LRW_350ms.txt'
ratio = 0.65


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
  # sub_folders = listdir_nohidden(img_feature_dir)
  train_list = listdir_nohidden(img_feature_dir)
  train_list.sort()
  # for each speaker
  for sub_folder in train_list:
    print(sub_folder)

    image_list = glob.glob(os.path.join(img_feature_dir, sub_folder) + '/*.jpg')
    print(os.path.join(img_feature_dir, sub_folder))
    image_list.sort(key = alphanum_key)


    
    train_sample_list = get_training_list(image_list)
    
    

    base_basename = '_'.join(os.path.basename(image_list[0]).split('_')[:-1])
    

    base_idx = 0
    for i in range(len(image_list)):
      # file audio file
      audio_file = image_list[i].replace('jpg', 'mat').replace('extract_faces', 'audio_features')

      cur_basename = '_'.join(os.path.basename(image_list[i]).split('_')[:-1])
      # print('cur_basename', cur_basename)

      if base_basename != cur_basename:
        base_idx = i
        base_basename = cur_basename
        # if cur_basename in test_sample_list:
          # f_test.write(os.path.abspath(image_list[i]) + '\n')

      if cur_basename in train_sample_list:
        f_train.write(os.path.abspath(image_list[base_idx]) + ' ' + os.path.abspath(image_list[i]) + ' ' + os.path.abspath(audio_file) + '\n')


  f_train.close()
  # f_test.close()


