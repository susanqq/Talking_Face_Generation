import os, glob
import numpy as np
from scipy.misc import imread, imresize, imsave
from scipy.io import loadmat
import scipy.io.wavfile as wav


def load_image(filename, image_size=112):
    image = imread(filename).astype(np.float32)
    # resize
    image = imresize(image, [image_size, image_size])
    # normalize
    image = (image*2.0/255.0-1.0).astype(np.float32)
    return image


def load_audio(audio_path):
    # load the mat file
    file = loadmat(audio_path)
    audio = np.transpose(file['mfcc'],(1,0))[1:,:]
    audio = np.expand_dims(audio, axis=-1)
    return audio