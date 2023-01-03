# -*- coding: utf-8 -*-
import logging
import pickle
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    def normalise(x, mean, std):
        return (x - mean) / std
        # return x

    filenames = ["train_0.npz", "train_1.npz", "train_2.npz", "train_3.npz", "train_4.npz"]

    images = []
    labels = []
    for fname in filenames:
        a = np.load(f"{input_filepath}/{fname}", allow_pickle=True)
        images_ = a['images']
        labels_ = a['labels']
        images.append(images_)
        labels.append(labels_)
    train_images = torch.from_numpy(np.concatenate(images))
    train_images = normalise(train_images, train_images.mean(dim=0), train_images.std(dim=0)+1e-10)
    train_labels = torch.from_numpy(np.concatenate(labels))

    test = np.load(f"{input_filepath}/test.npz", allow_pickle=True)
    test_images = torch.from_numpy(test['images'])
    test_images = normalise(test_images, train_images.mean(dim=0), train_images.std(dim=0)+1e-10)
    test_labels = torch.from_numpy(test['labels'])

    with open(output_filepath + "/corruptmnist_train.npz", "wb") as fp:
        pickle.dump((train_images, train_labels), fp)
    with open(output_filepath + "/corruptmnist_test.npz", "wb") as fp:
        pickle.dump((test_images, test_labels), fp)

    print("Done.")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
