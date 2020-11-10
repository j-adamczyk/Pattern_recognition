from concurrent.futures.thread import ThreadPoolExecutor
from io import BytesIO
import os

import cv2
import numpy as np
import requests


def get_img(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; "
                                 "Intel Mac OS X 10_11_6) "
                                 "AppleWebKit/537.36 (KHTML, like Gecko) "
                                 "Chrome/61.0.3163.100 Safari/537.36"}

        response = requests.get(url, headers=headers, timeout=2)
        if response.status_code == requests.codes.ok:
            return response
        else:
            return None
    except Exception as e:
        return None


def download_images_from_file(label):
    with open("links_" + label + ".txt") as file:
        URLs = file.readlines()

    URLs = [url.rstrip() for url in URLs]

    next_img_num = 0

    num_threads = min(len(URLs), 30)
    for i in range(0, len(URLs), 50):
        part = URLs[i:i + 50]
        with ThreadPoolExecutor(num_threads) as executor:
            results = [result for result
                       in executor.map(get_img, part)
                       if result is not None]

        for result in results:
            filename = str(next_img_num) + ".png"
            filepath = os.path.join("dataset", label, filename)
            img_bytes = BytesIO(result.content)
            img = cv2.imdecode(np.frombuffer(img_bytes.read(), np.uint8),
                               flags=1)

            if img is None:
                continue

            height, width = img.shape[:2]
            if height < width:
                new_height = 224
                new_width = int((new_height / height) * width)
            else:
                new_width = 224
                new_height = int((new_width / width) * height)

            img = cv2.resize(img,
                             dsize=(new_width, new_height),
                             interpolation=cv2.INTER_AREA)

            cv2.imwrite(filepath, img)
            next_img_num += 1


def create_csv(labels):
    labels.sort()

    text = "file label\n"

    for label_num, label in enumerate(labels):
        label_dir = os.path.join("dataset", label + "s")
        filenames = os.listdir(label_dir)
        filenames = [os.path.join(label_dir, filename)
                     for filename in filenames]
        for filename in filenames:
            # save numerical values for labels
            text += filename + " " + str(label_num) + "\n"

    with open("dataset.csv", "w") as file:
        file.write(text)


if __name__ == '__main__':
    #for file in ["cats", "cats", "cats"]:
    #    download_images_from_file(file)

    labels = ["cat", "dog", "owl"]
    create_csv(labels)

