#!/bin/env python
# -*- coding: utf8 -*-

"""Show images and labels for it."""

# from PIL import Image, ImageTk
import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_images(images, cols=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.

    cols (Default = 1): Number of columns in figure (number of rows is
                        set to np.ceil(n_images/float(cols))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images / float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()


def main():
    """Do the job"""
    images_dir = ("d:\\chukh\\Documents\\Texts\\Tyohaku\\2019\\ReconAI\\Recruitment_test\\"
                  "public-20190225T121409Z-001\\public\\test\\")
    images_ext = ".png"
    nums = np.array(np.arange(1, 301))  # Images are named as NN.png, where NN = 1...300
    left_labels_name = "test_labels_left.txt"
    right_labels_name = "test_labels_right.txt"
    print(images_dir, images_ext)
    print(nums)
    np.random.shuffle(nums)
    names = ["{}{}".format(num, images_ext) for num in nums[:3]]
    full_names = ["{}{}".format(images_dir, name) for name in names]
    print(names)
    print(full_names)
    images = [cv2.resize(cv2.imread(name), (200, 320)) for name in full_names]
    show_images(images, titles=names)


if __name__ == '__main__':
    main()
