import os, sys, glob
import re
import random
import cv2
from scipy.misc import imresize
from collections import defaultdict
import numpy as np

orig_training_file = "../preprocess_LRW/train_list_LRW_350ms_npy.txt"
new_training_file = "LRW_training_list_npy_128.txt"
spliter = '_'
IMAGE_RESIZE = 128
IMAGE_SIZE = 300
ratio = float(IMAGE_RESIZE)/IMAGE_SIZE

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



if __name__ == '__main__':
  r_file = open(orig_training_file, 'r')
  w_file = open(new_training_file, 'w')

  training_list = r_file.readlines() # image_path audio_path

  word_dict = {}

  for line in training_list:

    line = line.rstrip('\n')

    image_path, audio_path = line.split(' ')

    word = image_path.split('/')[-2]
    if word not in word_dict.keys():
      word_dict[word] = len(word_dict.keys())


    y1, y2, x1, x2 = 155, 264, 94, 203
    y1, y2, x1, x2 = ratio*y1, ratio*y2, ratio*x1, ratio*x2
    center_x = (x1+x2)*0.5
    center_y = (y1+y2)*0.5
    y1, y2, x1, x2 = int(center_y-20) , int(center_y+20), int(center_x-20), int(center_x+20)

    # image = np.load(image_path)[0]
    # print(image.shape)
    # image = imresize(image, [128,128])
    # roi = image[y1:y2, x1:x2]
    # cv2.imwrite('mouth_roi.jpg', roi)


    coordinate_str = ','.join(map(str, [x1,y1,x2,y2]))

    print(image_path + ' ' + audio_path + ' ' + coordinate_str + ' ' + str(word_dict[word]))
    w_file.write(image_path + ' ' + audio_path + ' ' + coordinate_str + ' ' + str(word_dict[word]) + '\n')


  r_file.close()
  w_file.close()





