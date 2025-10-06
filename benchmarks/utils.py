import os
import numpy as np
from numpy.fft import fft2, ifft2
import cv2
import time
import pandas
from PIL import Image, ImageOps


IMG_FILE_EXTENSIONS = ['.jpg', '.jpeg', '.png']


def load_img(filename):
    img_array_rgb = Image.open(filename)
    img_grayscale = ImageOps.grayscale(img_array_rgb)
    return np.array(img_grayscale)

def subpixel_shift(imagem, dx, dy, janela=None):
    h, w = imagem.shape
    imagem_f = imagem.astype(np.float64)

    if janela is not None:
        imagem_f *= janela

    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    FX, FY = np.meshgrid(fx, fy)
    fase = np.exp(-2j * np.pi * (FX * dx + FY * dy))
    I_fft = fft2(imagem_f)
    I_deslocada = ifft2(I_fft * fase)
    return np.real(I_deslocada)


def add_gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape)
    return img + noise


def add_salt_and_pepper(img, amount=90 / 100, s_vs_p=0.5):
    noisy = img.copy()
    num_salt = np.ceil(amount * img.size * s_vs_p)
    coords = (np.random.randint(0, img.shape[0], int(num_salt)),
              np.random.randint(0, img.shape[1], int(num_salt)))
    noisy[coords] = 255

    num_pepper = np.ceil(amount * img.size * (1. - s_vs_p))
    coords = (np.random.randint(0, img.shape[0], int(num_pepper)),
              np.random.randint(0, img.shape[1], int(num_pepper)))
    noisy[coords] = 0
    return noisy


def add_lens_blur(img, blur_kernel=(21, 21), feather=100):
    rows, cols = img.shape[:2]
    mask = np.zeros((rows, cols), dtype=np.float32)
    cv2.circle(mask, (cols // 2, rows // 2), min(rows, cols) // 2, 1, -1)

    # Smooth transition between sharp and blurred areas
    mask = cv2.GaussianBlur(mask, (101, 101), feather)

    blurred = cv2.GaussianBlur(img, blur_kernel, 0)
    blended = img * mask + blurred * (1 - mask)
    return blended


def add_full_blur(img, blur_kernel=(7, 7)):
    return cv2.GaussianBlur(img, blur_kernel, 0)

def create_datasets(data_root : str, overwrite: bool = False, verbose: bool = True) -> pandas.DataFrame:

    inspections = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
    ]

    if len(inspections) == 0:
        raise FileNotFoundError(f"No compatible files found at {data_root}")

    # Read images:
    t0 = time.time()
    for i, inspection in enumerate(inspections):
        data = {
            "inspection": [],
            "filename": [],
            "order": [],
            "img": [],
            "delta_x": [],
            "delta_y": []
        }

        imgs_path = data_root + inspection
        dataset_pkl_exist = os.path.exists(imgs_path)

        if (not overwrite) and dataset_pkl_exist:
            pass
        else:
            imgs_name = os.listdir(imgs_path)
            imgs_name = [f for f in imgs_name if f.lower().endswith(IMG_FILE_EXTENSIONS)]
            imgs_name.sort()

            # Read CSV dataset:
            try:
                calibration_data = pandas.read_csv(imgs_path + "/calibration_data.csv")
                px_per_mm = np.float32(calibration_data["px_p_mm"])
            except FileNotFoundError:
                px_per_mm = 1

            try:
                expected_path = pandas.read_csv(imgs_path + "/true_positions.csv")
            except FileNotFoundError:
                expected_path = np.array(len(imgs_name) * [0.0])

            data['order'].extend(np.array(range(len(imgs_name))))
            data['filename'].extend(imgs_name)
            data['inspection'].extend([inspection] * len(imgs_name))
            data['img'].extend([load_img(imgs_path + "/" + img_name) for img_name in imgs_name])
            data['delta_x'].extend(px_per_mm * expected_path)
            data['delta_y'].extend(px_per_mm * expected_path)

            dataset_df = pandas.DataFrame(data)
            dataset_df.to_pickle(imgs_path + "/dataset.pkl")

        if verbose:
            print(
                f"Dataset processing progress: {(i + 1)}/{len(inspections)} "
                f"({(i + 1) / len(inspections):.2%}) | "
                f"Elapsed time: {time.time() - t0:.1f} s",
                end="\r"
            )
    return dataset_df

def get_available_datasets(data_root: str, verbose: bool=False) -> list:
    valid_datasets = [
        name for name in os.listdir(data_root)
        if os.path.isdir(os.path.join(data_root, name))
           and os.path.exists(os.path.join(data_root, name, "dataset.pkl"))
    ]
    if verbose:
        print("List of valid datasets: \n", valid_datasets)
    return valid_datasets