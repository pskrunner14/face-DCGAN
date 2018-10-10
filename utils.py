import os

import cv2
import numpy as np

from tqdm import tqdm

# "Labelled Faces in the Wild" dataset
# Find dataset at: [http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz]
IMAGES_DIR = "datasets/lfw-deepfunneled"

def sample_noise_batch(batch_size, gaussian=True, emb_size=256):
    """ Returns batch of gaussian/uniform noise.

    Args:
        batch_size (int): Batch size for sampling noise.
        gaussian (bool, optional): Flag for sampling from gaussian noise distribution. If unspecified, defaults to `True`.
        emb_size (int, optional): Dimension of each noise sample in the array. If unspecified, defaults to 256.
    Returns:
        numpy.ndarray: Gaussian/uniformly sampled noise distribution.
    """
    if gaussian:
        return np.random.normal(loc=0.0, scale=1.0, size=(batch_size, emb_size)).astype('float32')
    return np.random.uniform(-1., 1., size=[batch_size, emb_size]).astype('float32')

def iterate_minibatches(inputs, batch_size=64, shuffle=False):
    """ Returns a mini-batch generator.

    Args:
        inputs (numpy.ndarray): Array of inputs (need to be atleast 2-dimensional).
        batch_size (int, optional): Batch size for sampling a mini-batch from the input distribution. If unspecified, defaults to 64.
        shuffle (bool, optional): Flag for randomly shuffling the data before generating mini-batches. If unspecified, defaults to `False`.

    Returns: 
        generator: Mini-batch generator for input images.
    """
    assert len(inputs.shape) >= 2, 'input needs to be atleast 2-dimensional.'
    
    if shuffle:
        indices = np.random.permutation(len(inputs))
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt]

def load_dataset(dx=80, dy=80, dimx=45, dimy=45):
    """
    Returns normalized and cropped array of images 
    in the `Labeled Faces in the Wild` dataset.

    Args:
        dx (int, optional): x co-ordinate to crop the images. If unspecified, defaults to 80.
        dy (int, optional): y co-ordinate to crop the images. If unspecified, defaults to 80.
        dimx (int, optional): Width dim of the images. If unspecified, defaults to 45.
        dimy (int, optional): Height dim of the images. If unspecified, defaults to 45.
    Returns:
        numpy.ndarray: Training data for the model.
        list of `int`: Shape of images in the training set.
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