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
        # size=(width, height)
        [cv2.resize(image,(new_size[1], new_size[0])) for image in image_array]
    )


#Saves a numpy array to file_path as a h5 file.
def save_images_to_h5py(images,file_path):
    h5f = h5py.File(file_path, "w")
    h5f.create_dataset("images", data=images)
    h5f.close()


#Loads a .h5 file from file_path
def load_images_from_hdf5(file_path):
    h5f = h5py.File(file_path, 'r')
    images = h5f['images'][:]
    h5f.close()
    return images


def load_and_convert_images_from_folder(folder_path, package_size, image_target_size, output_folder="datasets/images", resize=True):
    image_paths = os.listdir(folder_path)
    random.shuffle(image_paths)
    packages = (len(image_paths)//package_size) + 1
    print("Preprocessing images")
    for i in range(packages):
        print(f"{i}/{packages}")
        images = load_images_from_list(folder_path,image_paths[i*package_size:(i+1)*package_size])
        if resize:
            images = resize_images(images,image_target_size)
        save_images_to_h5py(images, output_folder + "/" + str(i)+".h5")


if __name__ == "__main__":
    # abstract
    #load_and_convert_images_from_folder("datasets/abstract_art_512", 64, (64,64), output_folder="datasets/abstract/64")
    #load_and_convert_images_from_folder("datasets/abstract_art_512", 128, (256,256), output_folder="datasets/abstract/256")
    #load_and_convert_images_from_folder("datasets/abstract_art_512", 128, (512,512), output_folder="datasets/abstract/512")
    #load_and_convert_images_from_folder("datasets/abstract_art_singleton", 2, (64,64), output_folder="datasets/abstract/singleton_64")
    # cats
    #load_and_convert_images_from_folder("datasets/cats_64", 256, (64,64), output_folder="datasets/cats/64")
    # plants
    load_and_convert_images_from_folder("datasets/plants_768x512", 64, (768,512), output_folder="datasets/plants/768x512", resize=False)





