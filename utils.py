import os

import cv2
import numpy as np

from tqdm import tqdm

IMAGES_DIR = "datasets/lfw-deepfunneled"    # Find dataset at: [http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz]

def sample_noise_batch(bsize, gaussian=True, emb_size=256):
    if gaussian:
        return np.random.normal(loc=0.0, scale=1.0, size=(bsize, emb_size)).astype('float32')
    return np.random.uniform(-1., 1., size=[bsize, emb_size]).astype('float32')

def iterate_minibatches(inputs, batchsize, shuffle=False):
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt]

def load_dataset(dx=80, dy=80, dimx=45, dimy=45):
    """
    Returns normalized and cropped array of images 
    in the `Labeled Faces in the Wild` dataset.

    Args:
        use_raw (bool, optional):
            Flag for using raw data or not. If unspecified, 
            defaults to `False`.
        dx (int, optional):
            x co-ordinate to crop the images. If unspecified, 
            defaults to 80.
        dy (int, optional):
            y co-ordinate to crop the images. If unspecified, 
            defaults to 80.
        dimx (int, optional):
            Width dim of the images. If unspecified, defaults 
            to 45.
        dimy (int, optional):
            Height dim of the images. If unspecified, defaults 
            to 45.
    
    Returns:
        numpy.ndarray:
            Training data for the model.
        list:
            Shape of images in the training set.
    """
    X = []

    folders = os.listdir(IMAGES_DIR)
    for folder in tqdm(folders, total=len(folders), desc='Preprocessing', leave=False):
        files = os.listdir(os.path.join(IMAGES_DIR, folder))
        for file in files:
            if not os.path.isfile(os.path.join(IMAGES_DIR, folder, file)) or not file.endswith(".jpg"):
                continue
            # preprocess image
            img = cv2.imread(os.path.join(IMAGES_DIR, folder, file))
            img = img[dy:-dy, dx:-dx]
            img = cv2.resize(img, (dimx, dimy))
            X.append(img)

    X = np.array(X)
    IMG_SHAPE = X.shape[1:]

    # normalize images
    X = (X.astype('float32') / 127.5) - 1.

    return X, IMG_SHAPE