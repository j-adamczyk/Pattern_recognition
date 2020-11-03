#!/usr/bin/env python

import cv2
import os
import numpy as np
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import matplotlib.pyplot as plt
from pylbfgs import owlqn


# Fraction to scale the original image
SCALE = 0.5

# Fraction of the scaled image to randomly sample
SAMPLE = 0.2

# Coeefficient for the L1 norm of variables (see OWL-QN algorithm)
ORTHANTWISE_C = 5

# File paths
ORIG_IMAGE_PATH = 'test/testimage.png'
A_FILE_PATH = 'test/a_matrix.npy'
B_FILE_PATH = 'test/b_vector.npy'


def dct2(x):
    return spfft.dct(
        spfft.dct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def idct2(x):
    return spfft.idct(
        spfft.idct(x.T, norm='ortho', axis=0).T, norm='ortho', axis=0)


def kron_rows(A, B, I, f=None):
    # find row indices of A and B
    ma, na = A.shape
    mb, nb = B.shape
    R = np.floor(I / mb).astype('int')  # A row indices of interest
    S = np.mod(I, mb)  # B row indices of interest

    # calculate kronecker product rows
    n = na * nb
    if f is None:
        K = np.zeros((I.size, n))

    for j, (r, s) in enumerate(zip(R, S)):
        row = np.multiply(
            np.kron(A[r, :], np.ones((1, nb))),
            np.kron(np.ones((1, na)), B[s, :])
            )
        if f is None:
            K[j, :] = row
        else:
            row.tofile(f)

    if f is None:
        return K


def progress(x, g, fx, xnorm, gnorm, step, k, ls):
    return 0


_image_dims = None  # track target image dimensions here
_ri_vector = None  # reference the random sampling indices here
_b_vector = None  # reference the sampled vector b here


def evaluate(x, g, step):
    # we want to return two things:
    # (1) the norm squared of the residuals, sum((Ax-b).^2), and
    # (2) the gradient 2*A'(Ax-b)

    # expand x columns-first
    x2 = x.reshape((_image_dims[1], _image_dims[0])).T

    # Ax is just the inverse 2D dct of x2
    Ax2 = idct2(x2)

    # stack columns and extract samples
    Ax = Ax2.T.flat[_ri_vector].reshape(_b_vector.shape)

    # calculate the residual Ax-b and its 2-norm squared
    Axb = Ax - _b_vector
    fx = np.sum(np.power(Axb, 2))

    # project residual vector (k x 1) onto blank image (ny x nx)
    Axb2 = np.zeros(x2.shape)
    Axb2.T.flat[_ri_vector] = Axb  # fill columns-first

    # A'(Ax-b) is just the 2D dct of Axb2
    AtAxb2 = 2 * dct2(Axb2)
    AtAxb = AtAxb2.T.reshape(x.shape)  # stack columns

    # copy over the gradient vector
    np.copyto(g, AtAxb)

    return fx


_A_matrix = None  # reference the dct matrix operator A here


def main(X, ri):
    global _b_vector, _A_matrix, _image_dims, _ri_vector

    # read image in grayscale, then downscale it
    ny, nx = X.shape

    # take random samples of image, store them in a vector b
    b = X.T.flat[ri].astype(float)  # important: cast to 64 bit

    # This method evaluates the objective function sum((Ax-b).^2) and its
    # gradient without ever actually generating A (which can be massive).
    # Our ability to do this stems from our knowledge that Ax is just the
    # sampled idct2 of the spectral image (x in matrix form).

    # save image dims, sampling vector, and b vector and to global vars
    _image_dims = (ny, nx)
    _ri_vector = ri
    _b_vector = np.expand_dims(b, axis=1)

    # perform the L1 minimization in memory
    Xat2 = owlqn(nx*ny, evaluate, progress, ORTHANTWISE_C)

    # transform the output back into the spatial domain
    Xat = Xat2.reshape(nx, ny).T  # stack columns
    Xa = idct2(Xat)

    # create images of mask (for visualization)
    mask = np.zeros(X.shape)
    mask.T.flat[ri] = 255
    Xm = 255 * np.ones(X.shape)
    Xm.T.flat[ri] = X.T.flat[ri]

    # display the result
    return Xm, Xa


def try_different_samples(filename):
    global ORIG_IMAGE_PATH, SAMPLE
    ORIG_IMAGE_PATH = filename
    Xorig = spimg.imread(ORIG_IMAGE_PATH, flatten=True, mode='L')
    X = spimg.zoom(Xorig, SCALE)

    ns = [0.1, 0.2, 0.3, 0.4, 0.5]
    errors = []
    for n in ns:
        SAMPLE = n
        ny, nx = X.shape
        k = round(nx * ny * SAMPLE)
        ri = np.random.choice(nx * ny, k, replace=False)  # random sample of indices
        mask, reconstructed = main(X, ri)

        error = np.abs(X - reconstructed).sum()
        errors.append(error)

        if n in [0.1, 0.5]:
            mask_filename = "mask_" + str(n).replace(".", "_") + "_" + filename
            rec_filename = "rec_" + str(n).replace(".", "_") + "_" + filename

            mask_filepath = os.path.join("cs_results", mask_filename)
            rec_filepath = os.path.join("cs_results", rec_filename)

            cv2.imwrite(mask_filepath, mask)
            cv2.imwrite(rec_filepath, reconstructed)

    plt.plot(ns, errors)
    plt.title(filename)
    plot_file = os.path.join("cs_results", "sample_size_plot_" + filename)
    plt.savefig(plot_file)
    plt.clf()


def try_smaller_scale(filename):
    global ORIG_IMAGE_PATH, SAMPLE, SCALE
    ORIG_IMAGE_PATH = filename
    Xorig = spimg.imread(ORIG_IMAGE_PATH, flatten=True, mode='L')
    SCALE = 0.2
    X = spimg.zoom(Xorig, SCALE)

    ns = [0.1, 0.2, 0.3, 0.4, 0.5]
    errors = []
    for n in ns:
        SAMPLE = n
        ny, nx = X.shape
        k = round(nx * ny * SAMPLE)
        ri = np.random.choice(nx * ny, k, replace=False)  # random sample of indices
        mask, reconstructed = main(X, ri)

        error = np.abs(X - reconstructed).sum()
        errors.append(error)

        if n in [0.1, 0.5]:
            mask_filename = "small_mask_" + str(n).replace(".", "_") + "_" + filename
            rec_filename = "small_rec_" + str(n).replace(".", "_") + "_" + filename

            mask_filepath = os.path.join("cs_results", mask_filename)
            rec_filepath = os.path.join("cs_results", rec_filename)

            cv2.imwrite(mask_filepath, mask)
            cv2.imwrite(rec_filepath, reconstructed)

    plt.plot(ns, errors)
    plt.title(filename)
    plot_file = os.path.join("cs_results", "small_sample_size_plot_" + filename)
    plt.savefig(plot_file)
    plt.clf()


def try_remove_fragment(filename):
    global ORIG_IMAGE_PATH
    ORIG_IMAGE_PATH = filename
    Xorig = spimg.imread(ORIG_IMAGE_PATH, flatten=True, mode='L')
    X = spimg.zoom(Xorig, SCALE)
    X[100:130, 100:130] = 255
    cv2.imwrite(os.path.join("cs_results", "removed_" + filename), X)

    ny, nx = X.shape
    k = round(nx * ny * SAMPLE)
    ri = np.random.choice(nx * ny, k, replace=False)  # random sample of indices
    mask, reconstructed = main(X, ri)

    mask_filename = "removed_mask_" + filename
    rec_filename = "removed_rec_" + filename

    mask_filepath = os.path.join("cs_results", mask_filename)
    rec_filepath = os.path.join("cs_results", rec_filename)

    cv2.imwrite(mask_filepath, mask)
    cv2.imwrite(rec_filepath, reconstructed)


def try_regular_sample(filename):
    global ORIG_IMAGE_PATH
    ORIG_IMAGE_PATH = filename
    Xorig = spimg.imread(ORIG_IMAGE_PATH, flatten=True, mode='L')
    X = spimg.zoom(Xorig, SCALE)

    ns = [0.1, 0.2, 0.3, 0.4, 0.5]
    errors_random = []
    errors_regular = []
    for n in ns:
        print(n)
        SAMPLE = n
        ny, nx = X.shape
        k = round(nx * ny * SAMPLE)
        ri_random = np.random.choice(nx * ny, k, replace=False)  # random sample of indices
        reconstructed_random = main(X, ri_random)[1]

        error_random = np.abs(X - reconstructed_random).sum()
        errors_random.append(error_random)

        ri_regular = np.array(list(range(0, k, (nx * ny) // k)))
        reconstructed_regular = main(X, ri_regular)[1]

        error_regular = np.abs(X - reconstructed_regular).sum()
        errors_regular.append(error_regular)

        if n in [0.1, 0.5]:
            rec_filename = "sampling_rand_" + str(n).replace(".", "_") + "_" + filename
            rec_filepath = os.path.join("cs_results", rec_filename)
            cv2.imwrite(rec_filepath, reconstructed_random)

            rec_filename = "sampling_reg_" + str(n).replace(".", "_") + "_" + filename
            rec_filepath = os.path.join("cs_results", rec_filename)
            cv2.imwrite(rec_filepath, reconstructed_regular)

    plt.plot(ns, errors_random, label="Random sampling")
    plt.plot(ns, errors_regular, label="Regular sampling")
    plt.title(filename)
    plt.legend(loc="upper right")
    plot_file = os.path.join("cs_results", "diff_sampling_plot_" + filename)
    plt.savefig(plot_file)
    plt.clf()


if __name__ == '__main__':
    for filename in ["img_building.png", "img_face.png", "img_pattern.png"]:
        #try_different_samples(filename)
        #try_smaller_scale(filename)
        #try_remove_fragment(filename)
        try_regular_sample(filename)
