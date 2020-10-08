import os
import cv2
import numpy as np
import random
from data import load_images_from_hdf5

def image_generator(batch_size, data_folder):
    current_length = 0
    current_sets = []
    paths = os.listdir(data_folder)
    paths = list(filter(lambda x: x[-3:] == ".h5", paths))
    while True:

        random.shuffle(paths)

        for path in paths:
            current_sets.append(load_images_from_hdf5(os.path.join(data_folder,path)))
            current_length += current_sets[-1].shape[0]
            #print(current_length)

            if current_length >= batch_size:
                last_set_length = current_sets[-1].shape[0]
                current_batch = np.concatenate(current_sets[:-1]+[current_sets[-1][:last_set_length+current_length-batch_size]])

                image_size = current_batch.shape[-2]
                current_batch.resize((batch_size,image_size,image_size,3))
                current_sets=[current_sets[-1][last_set_length-(current_length-batch_size):]]
                current_length = current_sets[0].shape[0]
                yield current_batch