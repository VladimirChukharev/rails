#!/bin/env python
# -*- coding: utf8 -*-

"""Show images and labels for it."""

# from PIL import Image, ImageTk
import os

import matplotlib.pyplot as plt
import numpy as np
import cv2


def show_images(images, rows=1, titles=None):
    """Display a list of images in a single figure with matplotlib.

    Parameters
    ---------
    images: List of np.arrays compatible with plt.imshow.
    rows=1: Number of columns in figure (number of cols is
                        set to np.ceil(n_images/float(rows))).

    titles: List of titles corresponding to each image. Must have
            the same length as titles.
    """
    assert ((titles is None) or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    figure = plt.figure()
    cols = np.ceil(n_images / float(rows))
    ax0 = None
    for indx, (image, title) in enumerate(zip(images, titles)):
        axes = figure.add_subplot(rows, cols, indx+1, sharey=ax0)
        if indx == 0:
            ax0 = axes
        else:
            plt.setp(axes.get_yticklabels(), visible=False)
        plt.imshow(image)
        axes.set_title(title)
    # figure.set_size_inches(np.array(figure.get_size_inches()) * n_images)
    plt.show()


def get_labels(images_dir, filename):
    """Read labels from file"""
    labels = np.loadtxt(os.path.join(images_dir, filename)).reshape((-1, 5, 2))
    return labels


def shuffle(length, images_dir, images_ext, scaling_factor, indexes, labels1, labels2):
    """Randomly shuffle indexes and return and length elements selected by indexes from each list"""
    np.random.shuffle(indexes)
    indxs = indexes[:length]
    names = ["{}{}".format(num + 1, images_ext) for num in indxs]
    full_names = ["{}{}".format(images_dir, name) for name in names]
    labels1 = np.array([(x, y) for (x, y) in [labels1[num][pnt] for num in indxs for pnt in range(5)]])
    labels1 = labels1.reshape((-1, 5, 2)) * scaling_factor
    labels2 = np.array([(x, y) for (x, y) in [labels2[num][pnt] for num in indxs for pnt in range(5)]])
    labels2 = labels2.reshape((-1, 5, 2)) * scaling_factor
    images = [cv2.resize(cv2.imread(name), None, None, scaling_factor, scaling_factor) for name in full_names]
    return indxs, names, full_names, labels1, labels2, images


def main():
    """Do the job"""
    show_images_number = 4
    scaling_factor = 1/5
    images_dir = ("d:\\chukh\\Documents\\Texts\\Tyohaku\\2019\\ReconAI\\Recruitment_test\\"
                  "public-20190225T121409Z-001\\public\\test\\")
    images_ext = ".png"
    left_labels_name = "test_labels_left.txt"
    right_labels_name = "test_labels_right.txt"
    red = (196, 0, 0)
    # green = (0, 196, 0)
    # blue = (0, 0, 196)

    nums = np.array(np.arange(300))  # Images are named as NN.png, where NN = 1...300
    left_labels = get_labels(images_dir, left_labels_name)
    right_labels = get_labels(images_dir, right_labels_name)
    nums, names, full_names, left_labels, right_labels, images = shuffle(
        show_images_number, images_dir, images_ext, scaling_factor,
        nums, left_labels, right_labels)
    for ind in range(show_images_number):
        polyline_l = left_labels[ind].astype(np.int32).reshape((-1, 2))
        polyline_r = right_labels[ind].astype(np.int32).reshape((-1, 2))
        # Draw red line segments 2 px wide through 5 points
        cv2.polylines(images[ind], [polyline_l, polyline_r], False, red, 2)
    show_images(images, titles=names)


if __name__ == '__main__':
    main()
