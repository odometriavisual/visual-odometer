import numpy as np
import os
from PIL import Image
import re


def get_img(i, data_root, rgb=False):
    image_list = os.listdir(data_root)
    # image_name = f"image{i:02d}.jpg"
    if not rgb:
        image_name = list(filter(lambda x: f"image{i:02d}_" in x, image_list))[0]
        rgb2gray = lambda img_rgb: img_rgb[:, :, 0] * .299 + img_rgb[:, :, 1] * .587 + img_rgb[:, :, 2] * .114
        myImage = Image.open(data_root + image_name)
        img_rgb = np.array(myImage)
        img_gray = rgb2gray(img_rgb)
        return img_gray
    else:
        image_name = list(filter(lambda x: f"image_{i:02d}." in x, image_list))[0]
        myImage = Image.open(data_root + image_name)
        img_rgb = np.array(myImage)
        return img_rgb

def sorted_alphanumeric(data):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(data, key=alphanum_key)

def get_imgs(n=None, data_root=None):
    image_list = os.listdir(data_root)
    image_list = sorted_alphanumeric(image_list)  # Ordena os nomes dos arquivos numericamente

    # First read a sample image to obtain its height and width:
    if n is None:
        n = len(image_list)
    image_name = image_list[0]
    for img_name in image_list:
        img_path = os.path.join(data_root, img_name)
        if os.path.isfile(img_path):
            with Image.open(img_path) as img:
                img_rgb = np.array(img)
                img_gray = rgb2gray(img_rgb)
                img_height, img_width = img_gray.shape
                imgs = np.zeros(shape=(n, img_height, img_width))
                for i in range(n):
                    img_name = image_list[i]
                    img_path = os.path.join(data_root, img_name)
                    with Image.open(img_path) as img:
                        img_rgb = np.array(img)
                        img_gray = rgb2gray(img_rgb)
                        imgs[i, :, :] = img_gray
                return imgs


def get_rgb_imgs(n=None, data_root=None):
    image_list = os.listdir(data_root)
    image_list = sorted_alphanumeric(image_list)  # Ordena os nomes dos arquivos numericamente

    if n is None:
        n = len(image_list)
    for img_name in image_list:
        img_path = os.path.join(data_root, img_name)
        if os.path.isfile(img_path):
            with Image.open(img_path) as img:
                img_rgb = np.array(img)
                img_height, img_width, _ = img_rgb.shape
                imgs_rgb = np.zeros(shape=(n, img_height, img_width, 3), dtype=np.uint8)
                for i in range(n):
                    img_name = image_list[i]
                    img_path = os.path.join(data_root, img_name)
                    with Image.open(img_path) as img:
                        img_rgb = np.array(img)
                        imgs_rgb[i, :, :, :] = img_rgb
                return imgs_rgb
def rgb2gray(img_rgb):
    return img_rgb[:, :, 0] * 0.299 + img_rgb[:, :, 1] * 0.587 + img_rgb[:, :, 2] * 0.114


def get_euler_data(data_root, filename="eul_data", n=995):
    euler_data = np.zeros(shape=(n, 3))
    with open(data_root + "/" + filename + '.txt', 'r') as f:
        for i, line in enumerate(f):
            corrected_line = line.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(',')
            if "None" in corrected_line:
                corrected_line = previous_line
            euler_data[i, :] = np.array([float(x) for x in corrected_line])
            previous_line = corrected_line
    return euler_data


def get_quat_data(data_root, filename="quat_data", n=995):
    quat_data = np.zeros(shape=(n, 4))
    with open(data_root + "/" + filename + '.txt', 'r') as f:
        for i, line in enumerate(f):
            corrected_line = line.replace("(", "").replace(")", "").replace(" ", "").replace("\n", "").split(',')
            if "None" in corrected_line:
                corrected_line = previous_line
            quat_data[i, :] = np.array([float(x) for x in corrected_line])
            previous_line = corrected_line
    return quat_data
