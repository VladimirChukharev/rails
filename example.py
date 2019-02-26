#!/bin/env python
# -*- coding: utf8 -*-

"""Recruitment test. Deep learn to recognize rails in images from video frames."""

import argparse
import os
import json
# import torch
# import keras
# from keras.callbacks import Callback


# These commands specify the output directory and two training directories for
# train and test data (see the input format in example.yaml)
output_folder = os.environ.get("VH_OUTPUTS_DIR", ".")
input_base_folder = os.environ.get("VH_INPUTS_DIR", ".")
train_folder = os.environ.get("TRAIN_DIR", os.path.join(input_base_folder, "train"))
test_folder = os.environ.get("TEST_DIR", os.path.join(input_base_folder, "test"))


def LogAsJSON(epoch, logs=None):
    logs = logs or {}
    meta = {
        "epoch": epoch,
        "loss": logs.get("loss"),
        "val_loss": logs.get("val_loss"),
    }
    print(json.dumps({key: value for (key, value) in meta.items() if value is not None}))


def do_train(*, epochs, batch_size):
    #
    #
    #   Insert your training code here
    #
    #
    pass


def main():
    # Parse arguments (See example.yaml)
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', default=10, type=int)
    ap.add_argument('--batch-size', default=10, type=int)
    args = ap.parse_args()

    # Train the model
    model = do_train(
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    weights_file = os.path.join(output_folder, "weights.h5")
    json_file = os.path.join(output_folder, "model.json")

    # Note:
    # All files saved to the output folder (VH_OUTPUTS_DIR) are shown in the
    # outputs -section of the Valohai interface

    # Saving example using Keras syntax
    print("Saving weights to", weights_file)
    model.save_weights(weights_file)
    print("Saving JSON to", json_file)
    with open(json_file, "w") as json_fp:
        json_fp.write(model.to_json())


if __name__ == '__main__':
    main()
