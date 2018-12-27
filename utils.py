#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 22:25:02 2018

@author: longer
"""

import os
import numpy as np
import scipy.misc
import config as cfg

def read_image(dir, height=128,width=256):
    paths = os.listdir(dir)
    paths = [os.path.join(dir, path) for path in paths]

    size = len(paths)
    all_images = np.zeros([size, height, width])
    for index, path in enumerate(paths):
        all_images[index] = (scipy.misc.imread(path).astype(np.float)/255. - 0.5) * 2
    all_images = all_images[:,:,:,np.newaxis]


    return all_images[:,:,:height, :], all_images[:,:,height:, :]


def batch_iter(sourceData, sourceLabel, batch_size=128, shuffle=True):

    data_size = sourceData.shape[0]
    label_size = sourceLabel.shape[0]
    assert data_size == label_size
    assert batch_size <= data_size

    num_batches_per_epoch = int(len(sourceData) / batch_size) + 1

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = sourceData[shuffle_indices]
        shuffled_label = sourceLabel[shuffle_indices]
    else:
        shuffled_data = sourceData
        shuffled_label = sourceLabel

    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index], shuffled_label[start_index:end_index]


def save_image(data_input, data_output, step):
    data = np.concatenate([data_input, data_output], axis=2)
    nums, width, hegiht, dims = data.shape
    assert dims == 1
    data.resize([nums, width, hegiht])
    data = (data + 1) / 2.0
    if not os.path.exists(cfg.path):
        os.mkdir(cfg.path)
    path = os.path.join(cfg.path, 'step'+str(step))
    if not os.path.exists(path):
        os.mkdir(path)
    for i in range(nums):
        image = data[i].reshape([width, hegiht])
        scipy.misc.imsave(os.path.join(path,str(i+1)+'.png'), image)

def get_all_image_path(dir):
    images_path = list()
    classes = os.walk(dir).__next__()[1]
    for c in classes:
        c_dir = os.path.join(dir, c)
        walk = os.walk(c_dir).__next__()
        for sample in walk[2]:
            if sample.endswith('.JPG'):
                images_path.append(os.path.join(c_dir, sample))
            if sample.endswith('.tif'):
                images_path.append(os.path.join(c_dir, sample))
    return images_path


def imread(image_path, half_size=64, image_size=128):

    image  = (scipy.misc.imread(image_path).astype(np.float)/255. - 0.5) * 2

    assert image.shape[0] >= image_size and image.shape[1] >= image_size
    midX = image.shape[0] // 2
    midY=  image.shape[1] // 2
    return image[midX-half_size:midX+half_size, midY-half_size:midY+half_size]



def save_image_output(data, image_path, out_dir):
    '''
    image_path: '/home/longer/Desktop/my_unet/data/train/D7/HB15147.tif'
    '''
    _, width, hegiht, dims = data.shape
    assert dims == 1
    data.resize([width, hegiht])
    data = (data + 1) / 2.0 * 255
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    image_name = image_path.split('/')[-1].split('.')[0] + '.JPG'
    image_class = image_path.split('/')[-2]
    path = os.path.join(out_dir, image_class)

    if not os.path.exists(path):
        os.mkdir(path)

    scipy.misc.imsave(os.path.join(path, image_name), data)


if __name__ == '__main__':
    '''
    data_input = np.zeros([10, 64,64,1])
    data_output = np.ones([10, 64, 64,1])
    step =1
    save_image(data_input, data_output, step)
    '''

    dir = '/home/longer/Desktop/my_unet/data/train'
    images_path = get_all_image_path(dir)
    print(images_path)
    image1 = scipy.misc.imread(images_path[0])
    #image1  = (scipy.misc.imread(images_path[0]).astype(np.float)/255. - 0.5) * 2
    print(image1)
