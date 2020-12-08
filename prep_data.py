from config import IMAGES_TO_CONVERT, PACKAGE_SIZE, DATA_FOLDER, IMG_SIZE
from data_tools.data import load_and_convert_images_from_folder

if __name__ == "__main__":
    # abstract
    load_and_convert_images_from_folder(IMAGES_TO_CONVERT, PACKAGE_SIZE, (IMG_SIZE,IMG_SIZE), output_folder=DATA_FOLDER)