import os, sys, glob
import re
import random
import cv2
from scipy.misc import imresize
from collections import defaultdict

orig_training_file = "/SLP_Extended/susan.s/Documents/Speech2Vid/data/TCD/merge_train_list.txt"
new_training_file = "TCD_training_list.txt"
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


if __name__ == '__main__':
  r_file = open(orig_training_file, 'r')
  w_file = open(new_training_file, 'w')

  training_list = r_file.readlines()
  # training_list = training_list[20000:20200]
  input_seq_list, gt_seq_list, audio_seq_list = load_seq_input(training_list)

  for i, gt_list in enumerate(gt_seq_list):
    print("process sequence : ", i)
    avg_x, avg_y = 0, 0

    for image_path in gt_list:
      landmark_path = image_path.replace('extract_faces', 'lip_landmarks').replace('.jpg', '.txt')
      landmark_file = open(landmark_path, 'r')
      landmark_list = landmark_file.readlines()
      landmark_file.close()

      left_x, left_y = landmark_list[48].rstrip('\n').split(',')
      right_x, right_y = landmark_list[54].rstrip('\n').split(',')

      left_x, left_y, right_x, right_y = int(left_x), int(left_y), int(right_x), int(right_y)

      mid_x = ratio*(left_x+right_x)/2.0
      mid_y = ratio*(left_y+right_y)/2.0

      # roi_x1, roi_x2, roi_y1, roi_y2 =  mid_x-0.5*width, mid_x+0.5*width, mid_y-0.62*height, mid_y+0.38*height

      avg_x += mid_x
      avg_y += mid_y

    # get averaged mouth region
    avg_x = int(avg_x/len(gt_list))
    avg_y = int(avg_y/len(gt_list))

    left = avg_x-20
    right = avg_x+20
    up = avg_y-24
    bottom = avg_y+16


    # for image_path in gt_list:
    #   image = cv2.imread(image_path)
    #   image = imresize(image, [128,128])
    #   mouth_roi = image[up:bottom, left:right]
    #   basename = os.path.basename(image_path)
    #   output_name = "temp/" + basename
    #   cv2.imwrite(output_name, mouth_roi)


    # write to training file
    image_list = input_seq_list[i]
    audio_list = audio_seq_list[i]

    for j in range(len(image_list)):
      w_file.write(image_list[j] + ' ' + gt_list[j] + ' ' + audio_list[j] + ' ' +  ','.join((str(left),str(up),str(right),str(bottom)))+'\n')


  r_file.close()
  w_file.close()





