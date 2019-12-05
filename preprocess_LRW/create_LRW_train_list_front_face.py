import numpy as np
import glob
import os, sys, re
import shutil
import cv2
import scipy.misc
import multiprocessing
from joblib import Parallel, delayed
from collections import defaultdict

landmark_dir = '/SLP_Extended_1_0/data/aivision/bbc_word/non_align/landmarks/train'
input_img_dir = '/SLP_Extended_1_0/data/aivision/bbc_word/non_align/input_image/train'
gt_img_dir = '/SLP_Extended_1_0/data/aivision/bbc_word/non_align/gt_image/train'
audio_dir = '/SLP_Extended/susan.s/lrw_data/audio_features/train'
len_range = [11, 20]
spliter = '_'
training_file = 'train_list_LRW_front_face.txt'

FILE_IN = open(training_file, 'w')


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
  folder_list = []
  for f in os.listdir(path):
      if not f.startswith('.'):
          folder_list.append(f)
  return folder_list

def run_each_folder(sub_folder):

  sub_path = os.path.join(landmark_dir, sub_folder)
  file_list = glob.glob(sub_path + '/*.txt')
  file_list.sort(key = alphanum_key)

  dic = defaultdict(list)
  for landmark_path in file_list:
    filedir, filename = os.path.split(landmark_path)
    word = os.path.basename(filedir)
    videoname = '_'.join(filename.split(spliter)[:-1])
    key = word +'/'+ videoname
    dic[key].append(landmark_path)


  for key, item in dic.iteritems():
    if len(item)<len_range[0] or len(item)>len_range[1]:
      continue

    item.sort(key = alphanum_key)
    count = 0
    for landmark_file in item:
      with open(landmark_file, 'r') as f:
        content = f.readlines()
      landmarks = []
      for line in content:
        line.rstrip('\n')
        [x, y] = line.split(',')
        pos = np.array([float(x),float(y)])
        landmarks.append(pos)
      landmarks = np.array(landmarks).astype(np.float32)


      # left eye avg position
      left_eye_pos = np.zeros(2)
      for ii in range(36, 42):
        left_eye_pos += landmarks[ii]
      left_eye_pos = left_eye_pos/6

      # righ eye avg position
      right_eye_pos = np.zeros(2)
      for ii in range(42,48):
        right_eye_pos += landmarks[ii]
      right_eye_pos = right_eye_pos/6

      # nose avg position
      nose_pos_x = 0
      for ii in range(30, 31):
        nose_pos_x += landmarks[ii][0]
      nose_pos_x = nose_pos_x/1

      # mouth avg position
      mouth_pos = np.zeros(2)
      for ii in range(48,61):
        mouth_pos += landmarks[ii]
      mouth_pos = mouth_pos/13

      # print(abs(left_eye_pos[0]+right_eye_pos[0]-2*nose_pos_x))
      if abs(left_eye_pos[0]+right_eye_pos[0]-2*nose_pos_x) < 33:
        count += 1
        # filename = os.path.basename(landmark_file).replace('.txt', '.jpg')
        # src_path = os.path.join(gt_img_dir, sub_folder, filename)
        # dst_path = os.path.join('temp', filename)
        # shutil.copyfile(src_path, dst_path)
    if count >= len_range[0] and count <= len_range[1]:
      print(item[0])
      for landmark_file in item:
        filename = os.path.basename(landmark_file).replace('.txt', '.jpg')
        input_img = os.path.join(input_img_dir, sub_folder, filename)
        gt_img = os.path.join(gt_img_dir, sub_folder, filename)
        input_audio = os.path.join(audio_dir, sub_folder, filename.replace('.jpg','.mat'))
        FILE_IN.write(input_img + ' ' + gt_img + ' ' + input_audio + '\n')





if __name__ == '__main__':

  sub_folders = listdir_nohidden(landmark_dir)
  sub_folders.sort()

  for sub_folder in sub_folders:
    run_each_folder(sub_folder)

  # num_cores = multiprocessing.cpu_count()
  # print('num of cores: ', num_cores)
  # Parallel(n_jobs=num_cores)(delayed(run_each_folder)(sub_folder) for sub_folder in sub_folders)

