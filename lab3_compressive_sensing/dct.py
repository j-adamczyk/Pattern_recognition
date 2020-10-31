import cv2
import numpy as np
from scipy.fft import dctn, idctn


def low_frequencies(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    low_width = width // 4
    low_height = height // 4
    dct_coef = dctn(img)
    dct_coef = dct_coef[:low_height, :low_width]
    img = idctn(dct_coef)
    img = cv2.resize(img,
                     dsize=(width, height),
                     interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("low_" + filename, img)


def medium_frequencies(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    low_width = 500
    low_height = 500
    dct_coef = dctn(img)
    dct_coef = dct_coef[:low_height, :low_width]
    img = idctn(dct_coef)
    img = cv2.resize(img,
                     dsize=(width, height),
                     interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("medium_" + filename, img)


def low_removed(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    dct_coef = dctn(img)
    dct_coef = dct_coef[30:, 30:]
    img = idctn(dct_coef)
    img = cv2.resize(img,
                     dsize=(width, height),
                     interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("low_removed_" + filename, img)


def rescale_coef(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    dct_coef = dctn(img)
    for i in range(dct_coef.shape[0]):
        for j in range(dct_coef.shape[1]):
            dist = (i ** 2 + j ** 2) ** (1 / 3)
            if not np.isclose(dist, 0):
                dct_coef[i, j] /= dist
    img = idctn(dct_coef)
    img = cv2.resize(img,
                     dsize=(width, height),
                     interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("rescaled_coef_" + filename, img)


def middle_frequencies(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    dct_coef = dctn(img)

    new_coef = np.zeros((height - 100, width - 10))
    height = (height - 150) // 2
    width = (width - 150) // 2
    new_coef[:height, :width] = dct_coef[:height, :width]

    common_height = min(dct_coef.shape[0], new_coef.shape[0])
    common_width = min(dct_coef.shape[1], new_coef.shape[1])

    new_coef[-height:, :common_width] = dct_coef[-height:, :common_width]
    new_coef[:common_height, -width:] = dct_coef[:common_height, -width:]

    img = idctn(new_coef)
    img = cv2.resize(img,
                     dsize=(width, height),
                     interpolation=cv2.INTER_LINEAR)
    cv2.imwrite("middle_" + filename, img)


def swap_dct_coef():
    img_building = cv2.imread("img_building.png", cv2.IMREAD_GRAYSCALE)
    img_face = cv2.imread("img_face.png", cv2.IMREAD_GRAYSCALE)
    img_pattern = cv2.imread("img_pattern.png", cv2.IMREAD_GRAYSCALE)

    dct_building = dctn(img_building)
    dct_face = dctn(img_face)
    dct_pattern = dctn(img_pattern)

    # swap first 50x50 building and pattern coefficients
    dct_bp = dct_building.copy()
    dct_bp[:50, :50] = dct_pattern[:50, :50]
    img_bp = idctn(dct_bp)
    cv2.imwrite("swapped_bp.png", img_bp)

    # swap first 50x50 face and building coefficients
    dct_fb = dct_face.copy()
    dct_fb[:50, :50] = dct_building[:50, :50]
    img_fb = idctn(dct_fb)
    cv2.imwrite("swapped_fb.png", img_fb)


if __name__ == '__main__':
    for filename in ["img_building.png",
                     "img_face.png",
                     "img_pattern.png"]:
        #low_frequencies(filename)
        #medium_frequencies(filename)
        #low_removed(filename)
        #rescale_coef(filename)
        #middle_frequencies(filename)
        pass
    swap_dct_coef()
