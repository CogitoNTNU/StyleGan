import os
import numpy as np
import random
from data_tools.data import load_images_from_hdf5

#Før man kan kjøre denne må man kjøre load_and_convert_from_folder fra data.py, for å få bildene på riktig format.
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

            if current_length >= batch_size:
                last_set_length = current_sets[-1].shape[0]
                current_batch = np.concatenate(current_sets[:-1]+[current_sets[-1][:last_set_length+current_length-batch_size]])
                width = current_batch.shape[-2]
                height = current_batch.shape[-3]
                current_batch.resize((batch_size,height,width,3))
                current_sets=[current_sets[-1][last_set_length-(current_length-batch_size):]]
                current_length = current_sets[0].shape[0]
                yield current_batch