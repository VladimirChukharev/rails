#!/bin/env python
# -*- coding: utf8 -*-

"""Show images and labels for it."""

# from PIL import Image, ImageTk
import os

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


def get_labels(images_dir, filename):
    """Read labels from file"""
    labels = np.loadtxt(os.path.join(images_dir, filename)).reshape((-1, 5, 2))
    return labels


def shuffle(length, indexes, images, labels1, labels2):
    """Randomly shuffle indexes and return and length elements selected by indexes from each list"""
    np.random.shuffle(indexes)
    indxs = indexes[:length]
    return (
        indxs,
        (images[ind] for ind in indxs),
        (labels1[ind] for ind in indxs),
        (labels2[ind] for ind in indxs),
    )

def main():
    """Do the job"""
    show_images_number = 4
    scaling_factor = 1/5
    images_dir = ("d:\\chukh\\Documents\\Texts\\Tyohaku\\2019\\ReconAI\\Recruitment_test\\"
                  "public-20190225T121409Z-001\\public\\test\\")
    images_ext = ".png"
    green = (0, 196, 0)
    red = (196, 0, 0)
    blue = (0, 0, 196)
    nums = np.array(np.arange(300))  # Images are named as NN.png, where NN = 1...300
    left_labels_name = "test_labels_left.txt"
    right_labels_name = "test_labels_right.txt"

    left_labels = get_labels(images_dir, left_labels_name)
    right_labels = get_labels(images_dir, right_labels_name)
    # print(nums)
    np.random.shuffle(nums)
    nums = nums[:show_images_number]
    names = ["{}{}".format(num+1, images_ext) for num in nums]
    full_names = ["{}{}".format(images_dir, name) for name in names]
    left_labels = np.array([(x, y) for (x, y) in [left_labels[num][pnt] for num in nums for pnt in range(5)]])
    left_labels = left_labels.reshape((-1, 5, 2)) * scaling_factor
    right_labels = np.array([(x, y) for (x, y) in [right_labels[num][pnt] for num in nums for pnt in range(5)]])
    right_labels = right_labels.reshape((-1, 5, 2)) * scaling_factor
    print(names)
    print(full_names)
    print(left_labels)
    print(left_labels[0])
    images = [cv2.resize(cv2.imread(name), None, None, scaling_factor, scaling_factor) for name in full_names]
    for ind in range(len(images)):
        polyline_l = left_labels[ind].astype(np.int32).reshape((-1, 2))
        polyline_r = right_labels[ind].astype(np.int32).reshape((-1, 2))
        print(ind, nums[ind], polyline_l)
        # Draw red line segments 2 px wide through 5 points
        cv2.polylines(images[ind], [polyline_l], False, red, 2)
        cv2.polylines(images[ind], [polyline_r], False, red, 2)
        # cv2.line(images[ind], (0, 0), (50, 50), (196, 0, 0), 1)
    show_images(images, titles=names)


if __name__ == '__main__':
    main()
