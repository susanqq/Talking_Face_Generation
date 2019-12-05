import os, sys, glob
import re
import random
import cv2
import dlib
from scipy.misc import imresize
from collections import defaultdict


orig_training_file = "/SLP_Extended/susan.s/Documents/Speech2Vid/DATA/filelist_shuffle_vox_filterAll_warp_input_more.txt"
new_training_file = "VOX_training_list_128.txt"
spliter = '_'
IMAGE_RESIZE = 128
IMAGE_SIZE = 300
ratio = float(IMAGE_RESIZE)/IMAGE_SIZE


predictor_path = "../preprocess_LRW/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.abspath(predictor_path))


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


def load_seq_input(training_list):
  seq_dic = defaultdict(list)
  for line in training_list:
    image_path, gt_path, audio_path = line.rstrip('\n').split(' ')
    basepath, filename = os.path.split(gt_path)
    celebname = os.path.basename(basepath)
    audioname = '_'.join(filename.split(spliter)[:-1])
    key = celebname +'/'+ audioname
    seq_dic[key].append([image_path, gt_path, audio_path])

  input_seq_list = []
  gt_seq_list = []
  audio_seq_list = []
  for key, items in seq_dic.items():
    input_seq_list.append([elem[0] for elem in items])
    gt_seq_list.append([elem[1] for elem in items])
    audio_seq_list.append([elem[2] for elem in items])
  return input_seq_list, gt_seq_list, audio_seq_list


def get_landmarks(im):
  rects = detector(im, 0)
  if len(rects)==0:
    return None
  max_rect = rects[0]
  s = [[p.x, p.y] for p in predictor(im, max_rect).parts()]

  return s

if __name__ == '__main__':
  r_file = open(orig_training_file, 'r')
  w_file = open(new_training_file, 'w')

  training_list = r_file.readlines()
  # training_list = training_list[20000:20200]
  input_seq_list, gt_seq_list, audio_seq_list = load_seq_input(training_list)

  for i, gt_list in enumerate(gt_seq_list):
    print("process sequence : ", i)


    left, right, up, bottom = [], [], [], []
    for image_path in gt_list:
      landmark_path = image_path.replace('extract_face_no_resize', 'lip_landmark_no_resize').replace('.jpg', '.txt')
      landmark_file = open(landmark_path, 'r')
      landmark_list = landmark_file.readlines()
      landmark_file.close()

      image = cv2.imread(image_path)
      landmarks = get_landmarks(image)

      if landmarks is None:
        continue

      left_x, left_y = landmarks[48]
      right_x, right_y = landmarks[54]

      mid_x = ratio*(left_x+right_x)/2.0
      mid_y = ratio*(left_y+right_y)/2.0

      left.append(int(mid_x-20))
      right.append(int(mid_x+20))
      up.append(int(mid_y-24))
      bottom.append(int(mid_y+16))

    if len(gt_list)!=len(left):
      print("skip: ", gt_list[0])
      continue


    # for j, image_path in enumerate(gt_list):
    #   image = cv2.imread(image_path)
    #   image = imresize(image, [IMAGE_RESIZE,IMAGE_RESIZE])
    #   # mouth_roi = image[up:bottom, left:right]
    #   mouth_roi = image[up[j]:bottom[j], left[j]:right[j]]
    #   basename = os.path.basename(image_path)
    #   output_name = "temp/" + basename
    #   cv2.imwrite(output_name, mouth_roi)


    # write to training file
    image_list = input_seq_list[i]
    audio_list = audio_seq_list[i]

    for j in range(len(image_list)):
      w_file.write(image_list[j] + ' ' + gt_list[j] + ' ' + audio_list[j] + ' ' +  ','.join((str(left[j]),str(up[j]),str(right[j]),str(bottom[j])))+'\n')



  r_file.close()
  w_file.close()





