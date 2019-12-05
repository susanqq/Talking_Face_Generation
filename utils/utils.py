import os, glob
import numpy as np
from scipy.misc import imread, imresize, imsave
import scipy.io.wavfile as wav
import h5py
from subprocess import call

def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def listdir_nohidden(path):
    return glob.glob(os.path.join(path, '*'))


def save_sample_images(images, save_path):
    size_frame = int(np.sqrt(len(images)))
    img_h, img_w = images.shape[2], images.shape[3]
    frame = np.zeros([img_h * size_frame, img_w * size_frame, 3])

    for ind, image in enumerate(images):
        if ind >= size_frame*size_frame:
            break
        ind_col = ind % size_frame
        ind_row = ind // size_frame
        frame[(ind_row * img_h):(ind_row * img_h + img_h), (ind_col * img_w):(ind_col * img_w + img_w), :] = np.transpose(image,(1,2,0))

    imsave(save_path, frame)


def get_wav_duration(audioname):
    #get the wav duration in second
    fname = os.path.abspath(audioname)
    fs, input_signal = wav.read(audioname)
    return len(input_signal)/float(fs)


def sort_filename(files):
    files_sort = sorted(files, key=lambda x:(
            '%s_%04d' % ('_'.join(x.split('/')[-1].split('.')[0].split('_')[:-1]),
                         int(x.split('/')[-1].split('.')[0].split('_')[-1]))
        ))
    return files_sort

def save_video(audio_duration, audio_test_file, images, save_dir):
    create_dir(save_dir)
    for idx, image in enumerate(images):
        image = np.transpose(image,(1,2,0))
        imsave(os.path.join(save_dir,'%03d.png'%(idx)), image)

    time_per_img = audio_duration/len(images)
    call(["ffmpeg", "-y", "-framerate", str(1.0/time_per_img), "-start_number", "0", "-i", os.path.join(save_dir,"%03d.png"), "-i", audio_test_file, "-c:v", "libx264", "-r", "25", "-pix_fmt", "yuv420p", "-c:a", "aac", "-strict", "experimental", "-shortest", os.path.join(save_dir,"./output.mp4")])



