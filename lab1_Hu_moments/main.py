from collections import OrderedDict
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import img_utils


def check_invariance(letter):
    img = img_utils.generate_letter(letter)
    img_translated_1 = img_utils.translate(img, axis=0, pixels=20)
    img_translated_2 = img_utils.translate(img, axis=1, pixels=-20)
    img_scaled_1 = img_utils.zoom(img, times=1.2)
    img_scaled_2 = img_utils.zoom(img, times=1.5)
    img_rotated_1 = img_utils.rotate(img, degrees=45)
    img_rotated_2 = img_utils.rotate(img, degrees=-90)

    hu_moments = OrderedDict()
    hu_moments["original"] = cv2.HuMoments(
        cv2.moments(img)).flatten()
    hu_moments["translated_1"] = cv2.HuMoments(
        cv2.moments(img_translated_1)).flatten()
    hu_moments["translated_2"] = cv2.HuMoments(
        cv2.moments(img_translated_2)).flatten()
    hu_moments["scaled_1"] = cv2.HuMoments(
        cv2.moments(img_scaled_1)).flatten()
    hu_moments["scaled_2"] = cv2.HuMoments(
        cv2.moments(img_scaled_2)).flatten()
    hu_moments["rotated_1"] = cv2.HuMoments(
        cv2.moments(img_rotated_1)).flatten()
    hu_moments["rotated_2"] = cv2.HuMoments(
        cv2.moments(img_rotated_2)).flatten()

    for i in range(7):
        ith_moment_vals = np.array([moments[i] for moments
                                    in hu_moments.values()])
        # get only moments for transforms
        ith_moment_vals = ith_moment_vals[1:]
        stdev = ith_moment_vals.std()

        original_val = hu_moments["original"][i]
        print(original_val, stdev)

    moment_1_vals = []
    for _ in range(1000):
        degrees = np.random.randint(low=-90, high=90)
        new_img = img_utils.rotate(img, degrees)
        moment_1_vals.append(cv2.HuMoments(cv2.moments(new_img))[0])

    moment_1_vals = np.array(moment_1_vals)
    print(letter, moment_1_vals.std())


def compare_transforms(letter):
    img = img_utils.generate_letter(letter)
    img_translated = img_utils.translate(img, axis=0, pixels=20)
    img_scaled = img_utils.zoom(img, times=1.2)
    img_rotated = img_utils.rotate(img, degrees=45)
    img_noise = img_utils.noise(img, low=-30, high=30)
    img_flipped = img_utils.flip(img, axis=0)

    Hu_moments = OrderedDict()
    Hu_moments["original"] = cv2.HuMoments(
        cv2.moments(img)).flatten()
    Hu_moments["translated"] = cv2.HuMoments(
        cv2.moments(img_translated)).flatten()
    Hu_moments["scaled"] = cv2.HuMoments(
        cv2.moments(img_scaled)).flatten()
    Hu_moments["rotated"] = cv2.HuMoments(
        cv2.moments(img_rotated)).flatten()
    Hu_moments["noise"] = cv2.HuMoments(
        cv2.moments(img_noise)).flatten()
    Hu_moments["flipped"] = cv2.HuMoments(
        cv2.moments(img_flipped)).flatten()

    x_labels = ["Original", "Translated", "Scaled", "Rotated", "Noise",
                "Flipped"]
    xs = list(range(6))

    for i in range(7):
        plt.title(f"Hu moment {i + 1} for {letter}")
        plt.xticks(xs, x_labels)
        ys = [moments[i] for moments in Hu_moments.values()]
        plt.scatter(xs, ys)
        plt.savefig(os.path.join("plots",
                                 "moment_" + str(i + 1) + "_" + letter))
        plt.clf()


def check_noise(letter):
    img = img_utils.generate_letter(letter)
    xs = list(range(1001))
    ys = [[] for _ in range(7)]

    for i in range(7):
        moments = cv2.HuMoments(cv2.moments(img))
        ys[i].append(moments[i])

    for _ in range(1000):
        img = img_utils.noise(img, low=-5, high=20)
        moments = cv2.HuMoments(cv2.moments(img))
        for i in range(7):
            ys[i].append(moments[i])

    for i in range(7):
        plt.plot(xs, ys[i])
        plt.title(f"Letter {letter}, moment {i + 1}")
        filepath = os.path.join("plots",
                                "noise_" + letter + "_" + str(i + 1) + ".png")
        plt.savefig(filepath)
        plt.clf()


def compare_letters_moments():
    P_img = img_utils.generate_letter("P")
    R_img = img_utils.generate_letter("R")
    W_img = img_utils.generate_letter("W")

    P_moments = cv2.HuMoments(cv2.moments(P_img)).flatten()
    R_moments = cv2.HuMoments(cv2.moments(R_img)).flatten()
    W_moments = cv2.HuMoments(cv2.moments(W_img)).flatten()

    xs = list(range(1, 8))

    plt.title("Comparison of moments values for letters")
    plt.scatter(xs, P_moments, c="red", label="P")
    plt.scatter(xs, R_moments, c="green", label="R")
    plt.scatter(xs, W_moments, c="blue", label="W")
    plt.legend()
    plt.savefig(os.path.join("plots", "comparison_1_7.png"))
    plt.clf()

    xs = list(range(2, 8))
    P_moments = P_moments[1:]
    R_moments = R_moments[1:]
    W_moments = W_moments[1:]

    plt.title("Comparison of moments values for letters")
    plt.scatter(xs, P_moments, c="red", label="P")
    plt.scatter(xs, R_moments, c="green", label="R")
    plt.scatter(xs, W_moments, c="blue", label="W")
    plt.legend()
    plt.savefig(os.path.join("plots", "comparison_2_7.png"))
    plt.clf()


def plot_2D():
    P_points = img_utils.make_letter_dataset("P")
    R_points = img_utils.make_letter_dataset("R")
    W_points = img_utils.make_letter_dataset("W")

    PCA_transform = PCA(n_components=2)
    P_points = PCA_transform.fit_transform(P_points)
    R_points = PCA_transform.fit_transform(R_points)
    W_points = PCA_transform.fit_transform(W_points)

    plt.title("Letter variations reduced to 2D")
    plt.scatter(P_points[:, 0], P_points[:, 1], c="red")
    plt.scatter(R_points[:, 0], R_points[:, 1], c="green")
    plt.scatter(W_points[:, 0], W_points[:, 1], c="blue")

    plt.savefig(os.path.join("plots", "after_PCA.png"))
    plt.clf()


def MNIST_dataset_experiment():
    X, y = img_utils.make_MNIST_dataset()

    # official train-test split for MNIST
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[-10000:], y[-10000:]

    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Hu moments + Random Forest accuracy:", round(accuracy * 100, 2))

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, cache=True)
    y = y.astype(int)
    # official train-test split for MNIST
    X_train, y_train = X[:60000], y[:60000]
    X_test, y_test = X[-10000:], y[-10000:]

    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Raw pixels + Random Forest accuracy:", round(accuracy * 100, 2))


if __name__ == '__main__':
    #compare_letters_moments()
    #plot_2D()
    for letter in ["P", "R", "W"]:
        #img = img_utils.generate_letter(letter, thickness=5)
        #cv2.imwrite(os.path.join("plots", letter + ".png"), img)
        #check_invariance(letter)
        #check_noise(letter)
        #compare_transforms(letter)
        pass

    #MNIST_dataset_experiment()

