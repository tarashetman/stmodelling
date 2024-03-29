import os
import torch
import torchvision
import torchvision.datasets as datasets

ROOT_DATASET = '/home/machine/PROJECTS/OTHER/fubel/stmodeling'


def return_kussaster(modality):
    filename_categories = 'kussaster/category.txt'

    filename_imglist_train = 'kussaster/train_videofolder.txt'
    filename_imglist_val = 'kussaster/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = 'kussaster'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = 'kussaster'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_kussaster_data(modality):
    filename_categories = 'kussaster/category.txt'
    filename_imglist_test = 'kussaster/test_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = 'kussaster'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = 'kussaster'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_test, root_data, prefix


def return_jester(modality):
    filename_categories = 'jester/category.txt'

    filename_imglist_train = 'jester/train_videofolder.txt'
    filename_imglist_val = 'jester/val_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = 'jester'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = 'jester'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_jester_data(modality):
    filename_categories = 'jester/category.txt'
    filename_imglist_test = 'jester/test_videofolder.txt'
    if modality == 'RGB':
        prefix = '{:05d}.jpg'
        root_data = 'jester'
    elif modality == 'RGBFlow':
        prefix = '{:05d}.jpg'
        root_data = 'jester'
    else:
        print('no such modality:' + modality)
        os.exit()
    return filename_categories, filename_imglist_test, root_data, prefix

def return_somethingv2(modality):
    filename_categories = 'something/category.txt'
    if modality == 'RGB':
        root_data = 'something'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    elif modality == 'Flow':
        root_data = 'something'
        filename_imglist_train = 'something/train_videofolder.txt'
        filename_imglist_val = 'something/val_videofolder.txt'
        prefix = '{:06d}.jpg'
    else:
        print('no such modality:' + modality)
    return filename_categories, filename_imglist_train, filename_imglist_val, root_data, prefix


def return_dataset(dataset, modality):
    dict_single = {'something': return_somethingv2, 'jester': return_jester, 'kussaster': return_kussaster}
    if dataset in dict_single:
        file_categories, file_imglist_train, file_imglist_val, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset ' + dataset)

    file_imglist_train = os.path.join(ROOT_DATASET, file_imglist_train)
    file_imglist_val = os.path.join(ROOT_DATASET, file_imglist_val)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_train, file_imglist_val, root_data, prefix


def return_data(dataset, modality):
    dict_single = {'jester': return_jester_data, 'kussaster': return_kussaster_data}
    if dataset in dict_single:
        file_categories, file_imglist_test, root_data, prefix = dict_single[dataset](modality)
    else:
        raise ValueError('Unknown dataset type' + dataset)

    file_imglist_test = os.path.join(ROOT_DATASET, file_imglist_test)
    file_categories = os.path.join(ROOT_DATASET, file_categories)
    with open(file_categories) as f:
        lines = f.readlines()
    categories = [item.rstrip() for item in lines]
    return categories, file_imglist_test, root_data, prefix