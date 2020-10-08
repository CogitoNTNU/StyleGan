import os
import cv2
import numpy as np
import h5py
import random


# Loops through a list of paths, loads the image, and converts the pixel values from [0,256] to [-1,1]
def load_images_from_list(original_path,path_list):

    images=[]

    for image_path in path_list:
        images.append(
            cv2.cvtColor(
                cv2.imread(os.path.join(original_path,image_path)),
                cv2.COLOR_BGR2RGB
            ) / 127.5-1
        )
    return np.array(images)

    #Resizes all the images in an array
def resize_images(image_array,new_size=(64,64)):
    return np.array(
        [cv2.resize(image,new_size) for image in image_array]
    )

def save_images_to_h5py(images,file_path):
    #Saves a numpy array to file_path as a h5 file.
    h5f = h5py.File(file_path, "w")
    h5f.create_dataset('images', data=images)
    h5f.close()

def load_images_from_hdf5(file_path):
    #
    h5f = h5py.File(file_path, 'r')
    images = h5f['images'][:]
    h5f.close()
    return images

def load_and_convert_images_from_folder(folder_path,package_size,image_target_size,filename="datasets/images"):
    image_paths = os.listdir(folder_path)
    for i in range(len(image_paths)//package_size+1):
        images = load_images_from_list(folder_path,image_paths[i*package_size:(i+1)*package_size])
        images = resize_images(images,image_target_size)
        save_images_to_h5py(images, filename+ "_" + str(i)+".h5")


if __name__ == "__main__":
    import timeit
    load_and_convert_images_from_folder("datasets/abstract_art_512",1,(64,64))
    #images =load_images_from_hdf5("datasets/images_0.h5")



