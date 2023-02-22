import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import keras
from keras.utils import Sequence
import os
import glob
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

DATASET_PATH = '/home/adrian/Datasets/iam_dataset/'
PADDING_TOKEN = 99
'''
The function creates the annotations file for train, val, test splits. If the
flag Debug is True, it uses a smaller dataset. 
'''
def create_splits(train = 0.85, val = 0.1, test = 0.05, debug = False):
    assert train + val + test == 1

    samples = []
    with open(DATASET_PATH + 'words.txt', 'r') as file:
        for sample in file.readlines():
            if sample[0] == "#" or sample.split(" ")[1] == 'err':
                continue
            samples.append(sample)
    
    if debug:
        samples = samples[0:1000]

    size = len(samples)
    print('Loaded {} samples.'.format(size))

    random.shuffle(samples)

    no_train = int(size * train)
    no_val = int(size * val)
    no_test = size - no_train - no_val
    
    train_samples = samples[0: no_train]
    val_samples = samples[no_train: no_train + no_val]
    test_samples = samples[no_train + no_val: size]

    print('Size of train dataset: ', no_train)
    print('Size of val dataset:', no_val)
    print('Size of test dataset:', no_test)

    def save_ann_file(split, samples, debug):
        if debug:
            split = 'mini-' + split

        with open(DATASET_PATH + split + '.txt', 'w') as file:
            for sample in samples:
                file.write(sample)
            file.close()

    save_ann_file('train', train_samples, debug)
    save_ann_file('val', val_samples, debug)
    save_ann_file('test', test_samples, debug)

'''
File naming convention: /f1/f1_f2/f1-f2-f3.png

Return: [paths], [labels]
'''
def get_annotations(samples):
    paths = []
    labels = []

    for sample in samples:
        split = sample.strip().split(" ")

        f1 = split[0]
        f2 = f1.split("-")[0]
        f3 = f1.split("-")[1]
        
        path = os.path.join(DATASET_PATH, 'words', f2, f2 + "-" + f3, f1 + '.png')
        
        if os.path.exists(path):
            paths.append(path)
            
            label = sample.split("\n")[0]
            label = label.split(" ")[-1].strip()

            labels.append(label)
        else:
            raise FileNotFoundError(path)
    
    return paths, labels

'''
Gets the vocabulary from labels.

Return: characters, max_length
'''
def get_vocabulary(labels):
    chars = set()
    max_length = 0

    for label in labels:
        label = label.split(" ")[-1].strip()

        for char in label:
            chars.add(char)
        
        length = len(label)
        if length > max_length:
            max_length = length

    chars = sorted(list(chars))

    return chars, max_length

def distortion_free_resize(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image

def get_dataset_parameters():
    char_str = '!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    char_array = [i for i in char_str]

    max_length = 21

    char_to_int = StringLookup(vocabulary=char_array, mask_token=None)
    int_to_char = StringLookup(
            vocabulary=char_to_int.get_vocabulary(), mask_token=None, invert=True)

    return char_array, char_to_int, int_to_char, max_length

class IAMDatasetGenerator(Sequence):
    def __init__(self, split, batch_size, size, debug = False):
        if debug:
            split = 'mini-' + split

        ann_path = DATASET_PATH + split + '.txt'

        samples = []
        with open(ann_path, 'r') as file:
            for sample in file.readlines():
                if sample[0] == "#" or sample.split(" ")[1] == 'err':
                    continue
                samples.append(sample)
        
        self.samples = samples
        self.size = len(samples)

        print('Loaded {} samples for {} split.'.format(self.size, split))

        self.paths, self.labels = get_annotations(samples)

        self.chars, self.char_to_int, self.int_to_char, self.max_length = get_dataset_parameters()
        '''
        self.chars, self.max_length = get_vocabulary(self.labels)

        # Mappings
        self.char_to_int = StringLookup(vocabulary=list(self.chars), mask_token=None)
        self.int_to_char = StringLookup(
            vocabulary=self.char_to_int.get_vocabulary(), mask_token=None, invert=True)
        '''

        self.batch_size = batch_size
        self.image_width, self.image_height = size
        self.split = split

    def display_sample(self, index):
        img = self.read_image(self.paths[index])
        label = self.labels[index]

        plt.title(label)
        plt.imshow(img, cmap='gray')

        plt.show() 

    def read_image(self, image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_png(img, 1)
        img = distortion_free_resize(img, (self.image_height, self.image_width))
        img = tf.cast(img, tf.float32) / 255.0

        return img

    def normalize_label(self, label):
        label = self.char_to_int(tf.strings.unicode_split(label, input_encoding="UTF-8"))

        length = tf.shape(label)[0]
        pad_amount = self.max_length - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=PADDING_TOKEN)

        return label

    def preprocess_image(self, image_path, img_size):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image


    def vectorize_label(self, label):
        label = self.char_to_int(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_length - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=PADDING_TOKEN)
        return label


    def process_images_labels(self, image_path, label):
        image = self.preprocess_image(image_path, (self.image_width, self.image_height))
        label = self.vectorize_label(label)
        return {"image": image, "label": label}

    def __len__(self):
        return self.size // self.batch_size



if __name__ == '__main__':
    create_splits(debug=False)

    size = (128, 32)
    debug = False
    train_datagen, val_datagen, test_datagen = IAMDatasetGenerator('train', 1, size), IAMDatasetGenerator('val', 1, size), IAMDatasetGenerator('test', 1, size)

    train_generator = IAMDatasetGenerator('train', 1, size, debug)
    val_generator = IAMDatasetGenerator('val', 1, size, debug)
    test_generator = IAMDatasetGenerator('test', 1, size, debug)

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = tf.data.Dataset.from_tensor_slices((train_generator.paths, train_generator.labels)).map(
        train_generator.process_images_labels, num_parallel_calls=AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_generator.paths, val_generator.labels)).map(
        val_generator.process_images_labels, num_parallel_calls=AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_generator.paths, test_generator.labels)).map(
        test_generator.process_images_labels, num_parallel_calls=AUTOTUNE)


    train = train_dataset.batch(1).cache().prefetch(AUTOTUNE)
    val = val_dataset.batch(1).cache().prefetch(AUTOTUNE)
    test = test_dataset.batch(1).cache().prefetch(AUTOTUNE)

    for i in range(0, 81986):
        print('Current', train_generator.paths[i])
        x = train_generator.process_images_labels(train_generator.paths[i], train_generator.labels[i])

    for sample in train:
        _ = True

    for sample in val:
        _ = True

    for sample in test:
        _ = True
    
    print('Done.')

    #Erased a01-117-05-02
    #       r06-022-03-05