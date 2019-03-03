#!/bin/env python
# -*- coding: utf8 -*-

"""Recruitment test. Deep learn to recognize rails in images from video frames."""

import argparse
import os
import json
import torch
import torch.cuda
import torch.nn as nn
import torch.optim
import torch.utils.data
# from torchvision import datasets, models, transforms
from torchvision import models, transforms
import numpy as np
from PIL import Image


# These commands specify the output directory and two training directories for
# train and test data (see the input format in example.yaml)
output_folder = os.environ.get("VH_OUTPUTS_DIR", ".")
input_base_folder = os.environ.get("VH_INPUTS_DIR", ".")
train_folder = os.environ.get("TRAIN_DIR", os.path.join(input_base_folder, "train"))
test_folder = os.environ.get("TEST_DIR", os.path.join(input_base_folder, "test"))
train_left_labels_name = "train_labels_left.txt"
train_right_labels_name = "train_labels_right.txt"
test_left_labels_name = "test_labels_left.txt"
test_right_labels_name = "test_labels_right.txt"


def pil_loader(path):
    """Use a loader from torchvision.datasets.folder"""
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def np_txt_loader(folder, target_files):
    """Combine labels from files for left and right rails as one array of N rows by 20 cols"""
    result0 = np.asarray(np.loadtxt(os.path.join(folder, target_files[0])), dtype=np.float32)
    len0, cols = result0.shape
    assert cols == 10
    result1 = np.asarray(np.loadtxt(os.path.join(folder, target_files[1])), dtype=np.float32)
    len1, cols = result1.shape
    assert cols == 10 and len0 == len1
    res = np.hstack((result0, result1))
    length, cols = res.shape
    assert cols == 20 and length == len0, "{}, {}, {}, {}.".format(cols, length, len0, len1)
    return res, length


class RailsDataset(torch.utils.data.Dataset):
    """
    Custom Dataset class with numbered images in a directory and targets as a file
    with a line corresponding to an image.
    """

    def __init__(self, folder, target_files, transform=None, target_transform=None, loader=pil_loader):
        # assert transform is None and target_transform is None, "Not yet implemented"
        self.folder = folder
        self.targets, self.length = np_txt_loader(folder, target_files)
        self.samples = [os.path.join(folder, "{}.png".format(ind+1)) for ind in range(self.length)]
        self.imgs = self.samples
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        print(self.targets.shape)

    def __getitem__(self, index):
        sample, target = self.loader(self.samples[index]), self.targets[index]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target

    def __len__(self):
        return self.length


def log_as_json(epoch, logs=None):
    """Write information to log journal in JSON format"""
    logs = logs or {}
    meta = {
        "epoch": epoch,
        "loss": logs.get("loss"),
        "val_loss": logs.get("val_loss"),
    }
    print(json.dumps({key: value for (key, value) in meta.items() if value is not None}))


def do_train(*, epochs, batch_size):
    """"Teach the neuron network using set of samples"""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    data_transforms = {
        'train': transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            # transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ]),
        'test': transforms.Compose([
            # transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize
        ]),
    }
    image_datasets = {
        'train': RailsDataset(train_folder, (train_left_labels_name, train_right_labels_name),
                              data_transforms['train']),
        'test': RailsDataset(test_folder, (test_left_labels_name, test_right_labels_name), data_transforms['test'])
    }
    dataloaders = {
        'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True,
                                             num_workers=4),
        'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size, shuffle=False, num_workers=4)
    }
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(pretrained=True).to(device)

    for param in model.parameters():
        param.requires_grad = False

    n_inputs = model.fc.in_features
    n_outputs = model.fc.out_features
    print(n_inputs, n_outputs)

    model.fc = nn.Sequential(
        nn.Linear(2048, 128),
        # nn.Conv2d(2048, 3, 3, 3),
        nn.ReLU(inplace=True),
        # nn.Conv2d(1, 1, 3),
        nn.Linear(128, 20)).to(device)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.MSELoss()
    # optimizer = torch.optim.Adam(model.fc.parameters())
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.ASGD(model.fc.parameters())

    print(sum(p.numel() for p in model.fc.parameters() if p.requires_grad))

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch+1, epochs))
        print('-' * 10)

        for phase in ['train', 'test']:
            for ind, par in enumerate(model.fc.parameters()):
                print(ind, par.shape)
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                print(labels)
                labels = labels.to(device)

                outputs = model(inputs)
                print(outputs, labels)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds = torch.max(outputs, 1)
                running_loss += loss.detach() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.float() / len(image_datasets[phase])

            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss.item(),
                                                        epoch_acc.item()))

    print(batch_size)
    log_as_json(epochs)

    return model


def main():
    """Do the job"""
    # Parse arguments (See example.yaml)
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', default=3, type=int)
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
    torch.save(model.state_dict(), 'models/pytorch/weights.h5')
    print("Saving JSON to", json_file)
    with open(json_file, "w") as json_fp:
        json_fp.write(model.to_json())


if __name__ == '__main__':
    main()
