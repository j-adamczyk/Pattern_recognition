import os
from typing import Union

import cv2
import numpy as np
from sklearn.datasets import fetch_openml


def generate_letter(letter: str,
                    font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale: Union[int, float] = 1,
                    thickness: int = 2) \
        -> np.ndarray:
    rows, cols = 100, 100

    # white image
    img = np.ones((rows, cols, 1), np.uint8) * 255

    # rescale font relative to image size
    font_scale = min(cols, rows) / (40 / font_scale)

    # make sure text is centered
    center_x, center_y = rows // 2, cols // 2
    size, _ = cv2.getTextSize(text=letter, fontFace=font, fontScale=font_scale,
                              thickness=thickness)
    top_left_x = center_x + size[1] // 2
    top_left_y = center_y - size[0] // 2

    # draw letter and return image
    img = cv2.putText(img=img, text=letter, org=(top_left_y, top_left_x),
                      fontFace=font, fontScale=font_scale, color=(0, 0, 0),
                      thickness=thickness, lineType=cv2.LINE_AA)

    return img


def rotate(image: np.ndarray,
           degrees: int = 90) \
        -> np.ndarray:
    # flip degrees, so they will follow normal convention (positive rotate
    # clockwise)
    degrees = - degrees

    # get image shape and flip - NumPy and OpenCV have different order
    image_shape = image.shape[1::-1]
    image_center = image_shape[0] // 2, image_shape[1] // 2

    # get rotation matrix to calculate rotation through matrix multiplication
    rotation_matrix = cv2.getRotationMatrix2D(center=image_center,
                                              angle=degrees,
                                              scale=1)

    # rotate with linear interpolation for better quality
    result = cv2.warpAffine(src=image,
                            M=rotation_matrix,
                            dsize=image_shape,
                            flags=cv2.INTER_LINEAR)

    # whiten out borders, since they turn black on the edges
    result[:, :10] = 255
    result[:, -10:] = 255
    result[:10, :] = 255
    result[-10:, :] = 255

    return result


def zoom(image: np.ndarray,
         times: Union[int, float] = 2) \
        -> np.ndarray:
    old_rows, old_cols = image.shape[:2]

    # resize image
    new_size = int(old_cols * times), int(old_rows * times)
    result = cv2.resize(src=image, dsize=new_size, fx=2, fy=2,
                        interpolation=cv2.INTER_LINEAR)

    # cut out the zoomed part (assumed to be central part)
    rows_center, cols_center = result.shape[0] // 2, result.shape[1] // 2
    rows_left = rows_center - old_rows // 2
    rows_right = rows_center + old_rows // 2
    cols_left = cols_center - old_cols // 2
    cols_right = cols_center + old_cols // 2
    result = result[rows_left:rows_right, cols_left:cols_right]

    return result


def translate(image: np.ndarray,
              axis: int = 0,
              pixels: int = 0) \
        -> np.ndarray:
    if axis == 0:
        translation_matrix = np.float32([[1, 0, 0],
                                         [0, 1, pixels]])
    else:
        translation_matrix = np.float32([[1, 0, pixels],
                                         [0, 1, 0]])

    result = cv2.warpAffine(src=image,
                            M=translation_matrix,
                            dsize=image.shape[1::-1])

    if axis == 0 and pixels > 0:
        result[:pixels + 1, :] = 255
    elif axis == 0:
        result[image.shape[0] - pixels:, :] = 255
    elif axis == 1 and pixels > 0:
        result[:, :pixels + 1] = 255
    else:
        result[:, image.shape[1] - pixels:] = 255

    return result


def noise(image: np.ndarray,
          low: int = -30,
          high: int = 30) \
        -> np.ndarray:
    noise_matrix = np.random.randint(low=low, high=high,
                                     size=image.shape)
    result = image + noise_matrix
    np.clip(a=result, a_min=0, a_max=255, out=result)
    result = result.astype(np.uint8)
    return result


def flip(image: np.ndarray,
         axis: int = 0) \
        -> np.ndarray:
    result = cv2.flip(src=image, flipCode=axis)
    return result


def make_letter_dataset(letter):
    """
    Dataset:
    - 4 fonts
    - 7 rotations with 45 degrees per rotation
    - 4 zoom levels (1.5, 2, 2.5, 3)
    - 4 translations with 30 pixels per direction (left, right, up, down)
    - 2 flips (axis 0 and 1)
    21 points in total.
    """
    matrix = np.zeros((21, 7))
    i = 0

    for font in [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
                 cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_TRIPLEX]:
        img = generate_letter(letter, font=font)
        moments = cv2.HuMoments(cv2.moments(img)).flatten()
        matrix[i] = moments
        i += 1

    for degrees in range(45, 360, 45):
        img = generate_letter(letter)
        img = rotate(img, degrees)
        moments = cv2.HuMoments(cv2.moments(img)).flatten()
        matrix[i] = moments
        i += 1

    for times in [1.5, 2, 2.5, 3]:
        img = generate_letter(letter)
        img = zoom(img, times)
        moments = cv2.HuMoments(cv2.moments(img)).flatten()
        matrix[i] = moments
        i += 1

    for pixels in [-30, 30]:
        for axis in [0, 1]:
            img = generate_letter(letter)
            img = translate(img, axis, pixels)
            moments = cv2.HuMoments(cv2.moments(img)).flatten()
            matrix[i] = moments
            i += 1

    for axis in [0, 1]:
        img = generate_letter(letter)
        img = flip(img, axis)
        moments = cv2.HuMoments(cv2.moments(img)).flatten()
        matrix[i] = moments
        i += 1

    return matrix


def make_MNIST_dataset():
    if "MNIST_moments.csv" in os.listdir() and "MNIST_y.csv" in os.listdir():
        X = np.genfromtxt("MNIST_moments.csv", delimiter=",")
        y = np.genfromtxt("MNIST_y.csv", delimiter=",")
        return X, y

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, cache=True)
    y = y.astype(int)
    X_moments = np.ndarray((X.shape[0], 7))
    for i in range(X.shape[0]):
        img = X[i].reshape((28, 28))
        X_moments[i] = cv2.HuMoments(cv2.moments(img)).flatten()

    np.savetxt("MNIST_moments.csv", X_moments, delimiter=",")
    np.savetxt("MNIST_y.csv", y, fmt="%i")

    return X, y
