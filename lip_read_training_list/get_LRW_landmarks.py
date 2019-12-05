import os, sys, glob
import re
import random
import cv2
from scipy.misc import imresize
from collections import defaultdict
import numpy as np
import dlib
import multiprocessing
from joblib import Parallel, delayed

predictor_path = "../preprocess_LRW/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.abspath(predictor_path))

gt_image_dir = '/SLP_Extended_1_0/data/aivision/bbc_word/non_align/gt_image/train'


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


def get_landmarks(im):
  rects = detector(im, 0)
  if len(rects)==0:
    return None
  max_rect = rects[0]
  s = [[p.x, p.y] for p in predictor(im, max_rect).parts()]
  return s


def run_each_folder(sub_folder):
  image_list = glob.glob(os.path.join(gt_image_dir, sub_folder) + '/*.jpg')

  for img_path in image_list:
    img = cv2.imread(img_path)

    landmarks = get_landmarks(img)

    landmark_path = img_path.replace('gt_image', 'landmarks').replace('.jpg', '.txt')

    if landmarks is None:
      if os.path.exists(landmark_path):
        print('remove existing file: ', landmark_path)
        os.remove(landmark_path)
        continue

    print(img_path, landmark_path)

    with open(landmark_path, 'w') as F:
      for landmark in landmarks:
        F.write(str(landmark[0]) + ',' + str(landmark[1]) + '\n' )


if __name__ == '__main__':

  folder_list = listdir_nohidden(gt_image_dir)

  num_cores = multiprocessing.cpu_count()

  Parallel(n_jobs=num_cores)(delayed(run_each_folder)(folder) for folder in folder_list)


